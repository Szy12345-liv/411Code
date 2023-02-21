# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import utils
import torch as t, torch.nn as nn, torch.nn.functional as tnnF, torch.distributions as tdist
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision as tv, torchvision.transforms as tr
import os
import sys
import argparse
#import ipdb
import numpy as np
import json
# Sampling
from tqdm import tqdm

from Util.datas import get_data
from Util.samples import get_sample_q
from Util.images import write_images
from models.ebm_models import get_model_and_buffer

t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3

##class torch.utils.tensorboard.writer.SummaryWriter(log_dir=runs/exp1, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')


def grad_norm(m):
    total_norm = 0
    for p in m.parameters():
        param_grad = p.grad
        if param_grad is not None:
            param_norm = param_grad.data.norm(2) ** 2
            total_norm += param_norm
    total_norm = total_norm ** (1. / 2)
    return total_norm.item()


def grad_vals(m):
    ps = []
    for p in m.parameters():
        if p.grad is not None:
            ps.append(p.grad.data.view(-1))
    ps = t.cat(ps)
    return ps.mean().item(), ps.std(), ps.abs().mean(), ps.abs().std(), ps.abs().min(), ps.abs().max()


def eval_classification(f, dload, device):
    corrects, losses = [], []
    for x_p_d, y_p_d in dload:
        x_p_d, y_p_d = x_p_d.to(device), y_p_d.to(device)
        logits = f.classify(x_p_d)
        loss = nn.CrossEntropyLoss(reduce=False)(logits, y_p_d).cpu().numpy()
        losses.extend(loss)
        correct = (logits.max(1)[1] == y_p_d).float().cpu().numpy()
        corrects.extend(correct)
    loss = np.mean(losses)
    correct = np.mean(corrects)
    return correct, loss


def checkpoint(f, buffer, num_gred, tag, args, device): ## checkpoint一般是把模型存下来
    f.cpu()
    ckpt_dict = {
        "model_state_dict": f.state_dict(),
        "replay_buffer": buffer,
        "num_gred_des": num_gred
    }
    t.save(ckpt_dict, os.path.join(args.save_dir, tag))
    f.to(device)

def cond_fid_is(f, replay_buffer, args, device, ratio=10000): ##这里的意思大概就是每个类选出分辨出来置信度最高的topk个图片，然后拿去算fid
    n_it = (replay_buffer.size(0) - 1) // 100
    all_y = []
    probs = []
    with t.no_grad():
        for i in tqdm(range(n_it+1)):
            if (i!=n_it):
                x = replay_buffer[i * 100: (i + 1) * 100].to(device)
            else:
                x = replay_buffer[i * 100: min((i + 1) * 100, replay_buffer.size(0))].to(device)
            logits = f.classify(x)
            y = logits.max(1)[1]
            prob = nn.Softmax(dim=1)(logits).max(1)[0]
            all_y.append(y)
            probs.append(prob)
    all_y = t.cat(all_y, 0)
    probs = t.cat(probs, 0)
    all_y = all_y.cpu()
    each_class = [replay_buffer[all_y == l] for l in range(args.n_classes)]
    each_class_probs = [probs[all_y == l] for l in range(args.n_classes)]

    new_buffer = []
    for c in range(args.n_classes):
        each_probs = each_class_probs[c]
        if ratio < 1:
            topk = int(len(each_probs) * ratio)
        else:
            topk = int(ratio)
        topk = min(topk, len(each_probs))
        topks = t.topk(each_probs, topk)
        index_list = topks[1].cpu()
        images = each_class[c][index_list]
        new_buffer.append(images)

    replay_buffer = t.cat(new_buffer, 0)
    replay_buffer = replay_buffer.cpu()
    from Task.eval_quality import eval_is_fid
    metrics = eval_is_fid((replay_buffer + 1) * 127.5, dataset=args.dataset, args=args)
    inc_score = metrics['inception_score_mean']
    std = metrics['inception_score_std']
    fid = metrics['frechet_inception_distance']
    print("Inception score of {} with std of {}".format(inc_score, std))
    print("FID of score {}".format(fid))
    return fid, inc_score

def main(args):
    utils.makedirs(args.save_dir)
    with open(f'{args.save_dir}/params.txt', 'w') as f:
        json.dump(args.__dict__, f)
    if args.print_to_log:
        sys.stdout = open(f'{args.save_dir}/log.txt', 'w')

    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)

    # datasets
    if args.sgld_initial:
        dload_train, dload_train_labeled, dload_valid, dload_test, dload_sample = get_data(args)
    else:
        dload_train, dload_train_labeled, dload_valid, dload_test = get_data(args)
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    if args.sgld_initial:
        sample_q, track_q, random_sampler = get_sample_q(args, device, dload_sample)
    else:
        sample_q, track_q, random_sampler = get_sample_q(args, device)
    if args.sgld_initial:
        f, replay_buffer, num_gred_des = get_model_and_buffer(args, device, sample_q, dload_sample)
    else:
        f, replay_buffer, num_gred_des = get_model_and_buffer(args, device, sample_q)
    sqrt = lambda x: int(t.sqrt(t.Tensor([x])))
    plot = lambda p, x: tv.utils.save_image(t.clamp(x, -1, 1), p, normalize=True, nrow=sqrt(x.size(0))) ##这个函数是干啥的

    # optimizer
    params = f.class_output.parameters() if args.clf_only else f.parameters()
    if args.optimizer == "adam":
        optim = t.optim.Adam(params, lr=args.lr, betas=[.9, .999], weight_decay=args.weight_decay)
    else:
        optim = t.optim.SGD(params, lr=args.lr, momentum=.9, weight_decay=args.weight_decay)

    best_valid_acc = 0.0
    cur_iter = 0 #每次反向传播后 cur_iter+1
    writer = SummaryWriter(log_dir=args.tensorboard_logdir)
    global_step=0
    for epoch in range(args.n_epochs):
        if epoch in args.decay_epochs:
            for param_group in optim.param_groups: #param_group是什么
                new_lr = param_group['lr'] * args.decay_rate #这里是梯度衰减
                param_group['lr'] = new_lr
                print("Decaying lr to {}".format(new_lr))
        for i, (x_p_d, _) in tqdm(enumerate(dload_train)): #x_p_d，_分别是什么
            
            global_step+=1

            if cur_iter <= args.warmup_iters: ##warmup_iters:预热学习率，先设置较小学习率使模型稳定
                lr = args.lr * cur_iter / float(args.warmup_iters)
                for param_group in optim.param_groups:
                    param_group['lr'] = lr

            x_p_d = x_p_d.to(device)
            x_lab, y_lab = dload_train_labeled.__next__()
            x_lab, y_lab = x_lab.to(device), y_lab.to(device)

            L = 0.
            E_p = 0.
            E_q = 0.

            if args.class_cond_p_x_sample:
                assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                x_q = sample_q(f, replay_buffer, num_gred_des, y=y_q) #这个y的意思其实是 sample p(x|y)
                ## track_q(f, replay_buffer, num_gred_des, writer, global_step)
            else:
                x_q = sample_q(f, replay_buffer, num_gred_des)  # sample from log-sumexp
                ## track_q(f, replay_buffer, num_gred_des, writer, global_step)

            fp_all = f(x_p_d) ## 这里是原样本(训练集) f返回的就是能量函数的值的相反数
            fq_all = f(x_q) ## 这里是采样的样本
            fp = fp_all.mean() ## 真实数据能量函数的平均值的相反数
            fq = fq_all.mean() ## 采样得到的样本能量函数的平均值的相反数

            E_p = -fp_all
            E_q = -fq_all
            l_p_x = -(fp - fq) ## MLE法
            writer.add_scalars("iter/ave_energy", {'true_data': -fp, 
                            'samples': -fq},
                            global_step=global_step)
            writer.add_scalar(tag="iter/energy_seperate", scalar_value=fp-fq,
                    global_step=global_step)
            if cur_iter % args.print_every == 0:
                print('P(x) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq, fp - fq))
            L += l_p_x + args.l2 * ((fp_all**2).mean() + (fq_all**2).mean()) ##这里的意思应该是 p(x)的权重，p(x)和p(y|x)设置不同的权重有利于正则化?


            writer.add_scalar(tag="iter/loss_generative", scalar_value=L,
                          global_step=global_step)
            
            if global_step % 100 == 0  :
                writer.add_histogram(tag="iter/Energy_negative", values=E_q, global_step=global_step)
                writer.add_histogram(tag="iter/Energy_positive", values=E_p, global_step=global_step)
            # break if the loss diverged...easier for poppa to run experiments this way
            if L.abs().item() > 1e8:
                print(global_step)
                print(L)
                track_q(f, replay_buffer, num_gred_des, writer, "Bao")
                print("BAD BOIIIIIIIIII")
                1/0

            optim.zero_grad()
            L.backward()
            optim.step()
            cur_iter += 1
            # 这个是输出图片的
            if cur_iter % args.print_every == 0:
                if args.plot_uncond: # generate class-conditional samples生成不带标签的样本
                    if args.class_cond_p_x_sample:
                        assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                        y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                        x_q = sample_q(f, replay_buffer, num_gred_des, y=y_q)
                    else:
                        x_q = sample_q(f, replay_buffer, num_gred_des)
                    E_q = -f(x_q)
                    plot('{}/x_q_{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q)
                    write_images('x_q_image_epoch{}_iter{:>06d}'.format(epoch,cur_iter), x_q, global_step, writer)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, num_gred_des, y=y)
                    E_q = -f(x_q_y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)
                    write_images('x_q_y_image_epoch{}_iter{:>06d}'.format(epoch,cur_iter), x_q_y, global_step, writer)
            
        if epoch % args.ckpt_every == 0: # 每10个epoch搞一个checkpoint
            checkpoint(f, replay_buffer, num_gred_des, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0: #每一个epoch evaluation一次
            f.eval() ## Sets the module in evaluation mode.
            print("Epoch {} Global_step {}".format(epoch, global_step))
            if args.plot_uncond: # generate class-conditional samples生成不带标签的样本

                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, num_gred_des, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer, num_gred_des,)

                plot('{}/x_q_{}.png'.format(args.save_dir, epoch), x_q)
                write_images('x_q_image_epoch{}'.format(epoch), x_q, epoch, writer)

                if  args.reinit_freq < 1:            
                    fid , isc = cond_fid_is(f, replay_buffer, args, device, args.ratio)
                    writer.add_scalars("epoch/fid_is", {'FID': fid, 
                                        'ISC': isc},
                                        epoch)
                
                buffer = random_sampler(f, replay_buffer.size(0), n_steps= args.n_steps)
                fid, isc = cond_fid_is(f, buffer, args, device, args.ratio)
                plot('{}/x_q_{}.png'.format(args.save_dir, epoch), buffer[0 : 64])
                write_images('x_q_y_image_epoch_random{}'.format(epoch),buffer[0 : 64] , epoch, writer)
                writer.add_scalars("epoch/fid_is_random", {'FID': fid, 
                                    'ISC': isc},
                                    epoch)
                
            if args.plot_cond:  # generate class-conditional samples
                y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                x_q_y = sample_q(f, replay_buffer, num_gred_des, y=y)

                plot('{}/x_q_{}.png'.format(args.save_dir, epoch), x_q_y)
                write_images('x_q_y_image_epoch{}'.format(epoch), x_q_y, epoch, writer)

                if  args.reinit_freq < 1:     
                    fid , isc = cond_fid_is(f, replay_buffer, args, device, args.ratio)
                    writer.add_scalars("epoch/fid_is", {'FID': fid, 
                                        'ISC': isc},
                                        epoch)
                
                buffer = random_sampler(f, replay_buffer.size(0), n_steps= args.n_steps)
                fid, isc = cond_fid_is(f, buffer, args, device, args.ratio)
                plot('{}/x_q_{}.png'.format(args.save_dir, epoch), buffer[0 : 64])
                write_images('x_q_y_image_epoch_random{}'.format(epoch),buffer[0 : 64] , epoch, writer)
                writer.add_scalars("epoch/fid_is_random", {'FID': fid, 
                                    'ISC': isc},
                                    epoch)
            f.train()
        checkpoint(f, replay_buffer, num_gred_des, "last_ckpt.pt", args, device)
    writer.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Energy Based Models and Shit")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "svhn", "cifar100"])
    parser.add_argument("--data_root", type=str, default="../data")
    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--decay_epochs", nargs="+", type=int, default=[160, 180],
                        help="decay learning rate by decay_rate at these epochs")
    parser.add_argument("--decay_rate", type=float, default=.3,
                        help="learning rate decay multiplier")
    parser.add_argument("--clf_only", action="store_true", help="If set, then only train the classifier")
    parser.add_argument("--labels_per_class", type=int, default=-1,
                        help="number of labeled examples per class, if zero then use all labels") ##这里是啥意思
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_iters", type=int, default=-1,
                        help="number of iters to linearly increase learning rate, if -1 then no warmmup")
    # loss weighting ##这些是啥东西
    parser.add_argument("--p_x_weight", type=float, default=1.)
    # regularization
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=3e-2,
                        help="stddev of gaussian noise to add to input, .03 works but .1 is more stable")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # network
    parser.add_argument("--norm", type=str, default=None, choices=[None, "norm", "batch", "instance", "layer", "act"],
                        help="norm to add to weights, none works fine")
    # EBM specific
    parser.add_argument("--n_steps", type=int, default=20,
                        help="number of steps of SGLD per iteration, 100 works for short-run, 20 works for PCD")
    parser.add_argument("--sgld_initial", default=False, action="store_true", help="If true, initialize sgld with trainning samples")
    parser.add_argument("--width", type=int, default=10, help="WRN width parameter")
    parser.add_argument("--depth", type=int, default=28, help="WRN depth parameter")
    parser.add_argument("--uncond", action="store_true", help="If set, then the EBM is unconditional")
    parser.add_argument("--class_cond_p_x_sample", action="store_true",
                        help="If set we sample from p(y)p(x|y), othewise sample from p(x),"
                             "Sample quality higher if set, but classification accuracy better if not.")
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--reinit_freq", type=float, default=.05)
    parser.add_argument("--sgld_lr", type=float, default=1.0)
    parser.add_argument("--sgld_std", type=float, default=1e-2)
    # logging + evaluation
    parser.add_argument("--save_dir", type=str, default='./experiment')
    parser.add_argument("--tensorboard_logdir", type=str, default='./runs')
    parser.add_argument("--ckpt_every", type=int, default=10, help="Epochs between checkpoint save")
    parser.add_argument("--eval_every", type=int, default=1, help="Epochs between evaluation")
    parser.add_argument("--print_every", type=int, default=100, help="Iterations between print")
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--print_to_log", action="store_true", help="If true, directs std-out to log file")
    parser.add_argument("--plot_cond", action="store_true", help="If set, save class-conditional samples")
    parser.add_argument("--plot_uncond", action="store_true", help="If set, save unconditional samples")
    parser.add_argument("--n_valid", type=int, default=5000)
    parser.add_argument("--ratio", type=float, default=10000)
    # random seed
    parser.add_argument("--seed",type=int, default=1234)
    args = parser.parse_args()
    args.n_classes = 100 if args.dataset == "cifar100" else 10
    main(args)