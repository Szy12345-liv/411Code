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
import wideresnet
import json
# Sampling
from tqdm import tqdm
t.backends.cudnn.benchmark = True
t.backends.cudnn.enabled = True
seed = 1
im_sz = 32
n_ch = 3

##class torch.utils.tensorboard.writer.SummaryWriter(log_dir=runs/exp1, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='')

class DataSubset(Dataset):
    def __init__(self, base_dataset, inds=None, size=-1):
        self.base_dataset = base_dataset
        if inds is None:
            inds = np.random.choice(list(range(len(base_dataset))), size, replace=False)
        self.inds = inds

    def __getitem__(self, index):
        base_ind = self.inds[index]
        return self.base_dataset[base_ind]

    def __len__(self):
        return len(self.inds)


class F(nn.Module): ##纯生成模型
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(F, self).__init__()
        self.f = wideresnet.Wide_ResNet(depth, width, norm=norm, dropout_rate=dropout_rate)
        self.energy_output = nn.Linear(self.f.last_dim, 1)
        self.class_output = nn.Linear(self.f.last_dim, n_classes)

    def forward(self, x, y=None):
        penult_z = self.f(x)
        return self.energy_output(penult_z).squeeze()

    def classify(self, x):
        penult_z = self.f(x)
        return self.class_output(penult_z).squeeze()


class CCF(F):  ##生成+判别
    def __init__(self, depth=28, width=2, norm=None, dropout_rate=0.0, n_classes=10):
        super(CCF, self).__init__(depth, width, norm=norm, dropout_rate=dropout_rate, n_classes=n_classes)

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1) #返回x的概率
        else:
            return t.gather(logits, 1, y[:, None]) #返回(x,y)的概率，y是一个数组(代表我有用到的label)


def cycle(loader):
    while True:
        for data in loader:
            yield data


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


def init_random(args, bs,  dload_sample = None):
    if args.sgld_initial:
        return t.cat([dload_sample.__next__()[0] for i in range(bs)],dim=0)
    else: 
        return t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1) ## bs是batch_size比如一组64张图? n_ch*im_sz*im_sz=3*32*32


def get_model_and_buffer(args, device, sample_q, dload_sample=None): ##这个replay buffer里面存的是啥东西(是生成图片结果的展示吗), sample_q应该是一个采样器
    model_cls = F if args.uncond else CCF
    f = model_cls(args.depth, args.width, args.norm, dropout_rate=args.dropout_rate, n_classes=args.n_classes) ## depth和width应该是wideresnet的深度和每层神经元的数量，norm应该是batch norm的方法
    if not args.uncond:
        assert args.buffer_size % args.n_classes == 0, "Buffer size must be divisible by args.n_classes"
    if args.load_path is None: ##还没训练
        # make replay buffer
        replay_buffer = init_random(args, args.buffer_size, dload_sample)
        shape = (args.buffer_size,)
        num_gred_des = t.zeros(shape)

    else: ##训完了，把参数导过来
        print(f"loading model from {args.load_path}")
        ckpt_dict = t.load(args.load_path) ##训练完的模型参数
        f.load_state_dict(ckpt_dict["model_state_dict"]) ##把参数导入模型
        replay_buffer = ckpt_dict["replay_buffer"]
        num_gred_des = ckpt_dict["num_gred_des"]

    f = f.to(device)
    return f, replay_buffer, num_gred_des 


def get_data(args):
    ##对dataset进行裁剪翻转等预处理
    if args.dataset == "svhn":
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz),
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)]
        )
    else:
        transform_train = tr.Compose(
            [tr.Pad(4, padding_mode="reflect"),
             tr.RandomCrop(im_sz), ##先填充再裁剪，随机crop
             tr.RandomHorizontalFlip(), ##水平翻转
             tr.ToTensor(),
             tr.Normalize((.5, .5, .5), (.5, .5, .5)),
             lambda x: x + args.sigma * t.randn_like(x)] ##这里加高斯噪声？
        )
    transform_test = tr.Compose(
        [tr.ToTensor(),
         tr.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + args.sigma * t.randn_like(x)]
    )
    def dataset_fn(train, transform):
        if args.dataset == "cifar10":
            return tv.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train) ##这里data_root这个参数在命令里没查到？
        elif args.dataset == "cifar100":
            return tv.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)
        else:
            return tv.datasets.SVHN(root=args.data_root, transform=transform, download=True,
                                    split="train" if train else "test")

    # get all training inds
    full_train = dataset_fn(True, transform_train) #先把预处理好的数据集下载下来
    print("full_data_length:{}".format(len(full_train)))
    all_inds = list(range(len(full_train))) #就是一个从0到data总长度减一的数组
    # set seed
    np.random.seed(args.seed)
    # shuffle
    np.random.shuffle(all_inds) #随机打乱 SGD
    # seperate out validation set
    if args.n_valid is not None:
        valid_inds, train_inds = all_inds[:args.n_valid], all_inds[args.n_valid:] #分割验证集训练集
    else:
        valid_inds, train_inds = [], all_inds #不分割
    train_inds = np.array(train_inds)
    train_labeled_inds = []
    other_inds = []
    train_labels = np.array([full_train[ind][1] for ind in train_inds]) #把训练集的标签拿出来
    if args.labels_per_class > 0: ##这里的意思应该是如果对每个class的样本数量有要求，就每个class的样本取出这么多个
        for i in range(args.n_classes): #n_classes就是数字的种类数，这里等于10
            print(i)
            train_labeled_inds.extend(train_inds[train_labels == i][:args.labels_per_class])
            other_inds.extend(train_inds[train_labels == i][args.labels_per_class:])
    else:
        train_labeled_inds = train_inds

    dset_train = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_sample = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_inds)
    dset_train_labeled = DataSubset(
        dataset_fn(True, transform_train),
        inds=train_labeled_inds)
    dset_valid = DataSubset(
        dataset_fn(True, transform_test),
        inds=valid_inds)
    dload_train = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_sample = DataLoader(dset_sample, batch_size=1, shuffle=True, num_workers=4, drop_last=True)
    dload_sample = cycle(dload_sample)
    dload_train_labeled = DataLoader(dset_train_labeled, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    dload_train_labeled = cycle(dload_train_labeled)
    dset_test = dataset_fn(False, transform_test)
    dload_valid = DataLoader(dset_valid, batch_size=100, shuffle=False, num_workers=4, drop_last=False) ##num_workers的意思是开启多少个子线程加载数据
    dload_test = DataLoader(dset_test, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
    ## 现在还没搞懂这俩dload_train, dload_train_labeled分别在训什么
    if args.sgld_initial:
        return dload_train, dload_train_labeled, dload_valid,dload_test,dload_sample
    else:
        return dload_train, dload_train_labeled, dload_valid,dload_test    


def get_sample_q(args, device, dload_sample=None):
    def sample_p_0(replay_buffer, num_gred_des, bs, y=None): ##bs就是 batch_size
        if len(replay_buffer) == 0:
            return init_random(args, bs, dload_sample), [], t.zeros(args.buffer_size)
        buffer_size = len(replay_buffer) if y is None else len(replay_buffer) // args.n_classes ##每个class均分buffer
        inds = t.randint(0, buffer_size, (bs,)) ##第一个参数是最小值，第二个参数是最大值-1，第三个参数是返回的随机tensor的shape
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds ##y.cpu是什么东西, y.cpu就是把y往cpu里复制一份，然后返回y，详见pytorch.org
            ##这里就是把inds变成符合标签的那一段值，上面那里inds的值是[0,buffer_size)的
            assert not args.uncond, "Can't drawn conditional samples without giving me y"
        buffer_samples = replay_buffer[inds] ##从y所属的buffer中按inds随机取出一些samples
        buffer_nums = num_gred_des[inds] ##表示replay buffer中的样本经过的梯度下降的次数
        random_samples = init_random(args, bs, dload_sample)
        rand = (t.rand(bs) < args.reinit_freq).float()
        choose_random = rand[:, None, None, None] ##torch.rand 返回了一个长度为bs的，每个数字为[0,1)均匀随机数的tensor， 然后扩展为四维数组
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples ##以0.05的概率选random_samples里面的样本，0.95的概率选buffer里面的样本
        nums = rand * 1 + (1 - rand) * (buffer_nums + 1)
        return samples.to(device), inds, nums

    def sample_q(f, replay_buffer, num_gred_des, y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0) ## y是一组标签， y.size(0)就是标签的数量
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds, nums = sample_p_0(replay_buffer, num_gred_des, bs=bs, y=y) ## 拿到样本和它们在buffer中的位置
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k) ##这里sgld的lr和std不应该是根号关系吗？为何可以自己调整噪声
        f.train() ##这是什么意思 Sets the module in training mode.
        final_samples = x_k.detach() ##采出来的样本不用求导了
        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu() ##
            num_gred_des[buffer_inds] = nums
        return final_samples
    def track_q(f, replay_buffer, num_gred_des, writer, global_step,  y=None, n_steps=args.n_steps):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        f.eval()
        # get batch size
        bs = args.batch_size if y is None else y.size(0) ## y是一组标签， y.size(0)就是标签的数量
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds, nums = sample_p_0(replay_buffer, num_gred_des, bs=bs, y=y) ## 拿到样本和它们在buffer中的位置
        x_k = t.autograd.Variable(init_sample, requires_grad=True)
        # sgld
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k) ##这里sgld的lr和std不应该是根号关系吗？为何可以自己调整噪声
            f_all = f(x_k)
            writer.add_scalars("step" + str(global_step) + "track" + "/energy", {'sample0': -f_all[0], 
                                    'sample1': -f_all[1],
                                    'sample2': -f_all[2]},
                                    k)
        x_k=t.clamp(x_k,-1,1)
        writer.add_images("step" + str(global_step) + "track" + "/image", x_k[0:3]*0.5+0.5, global_step)
        # 随机噪声
        random_samples = t.FloatTensor(3, n_ch, im_sz, im_sz).uniform_(-1, 1)
        random_samples = random_samples.to(device)
        x_k = t.autograd.Variable(random_samples, requires_grad=True)
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k) ##这里sgld的lr和std不应该是根号关系吗？为何可以自己调整噪声
            f_all = f(x_k)
            #print(k)
            writer.add_scalars("step" + str(global_step) + "track_random" + "/energy" , {'sample0': -f_all[0], 
                                    'sample1': -f_all[1],
                                    'sample2': -f_all[2]},
                                    k)
        x_k=t.clamp(x_k,-1,1)
        writer.add_images("step" + str(global_step) + "track_random" + "/image", x_k.cpu()*0.5+0.5, global_step)
        f.train() ##这是什么意思 Sets the module in training mode.
        final_samples = x_k.detach() ##采出来的样本不用求导了
        return final_samples
    return sample_q, track_q


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
    print(replay_buffer.shape)
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
        sample_q, track_q = get_sample_q(args, device, dload_sample)
    else:
        sample_q, track_q = get_sample_q(args, device)
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
    epoch=0
    if args.plot_uncond: # generate class-conditional samples生成不带标签的样本
        if args.class_cond_p_x_sample:
            assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
            y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
            x_q = sample_q(f, replay_buffer, num_gred_des, y=y_q)
        else:
            x_q = sample_q(f, replay_buffer, num_gred_des,)
        print("----------------------------")
        print(x_q.type())
        x_q = x_q.cpu()
        print(x_q.type())
        print("----------------------------")
        print(replay_buffer.type())
        print("----------------------------")
        writer.add_images('x_q_image_epoch{}'.format(epoch), x_q, epoch)
        fid , isc = cond_fid_is(f, x_q, args, device, args.ratio)
        writer.add_scalars("epoch/fid_is", {'FID': fid, 
                            'ISC': isc},
                            epoch)
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
            L_fp = 0.
            L_fq = 0.
            E_p = 0.
            E_q = 0.
            if args.p_x_weight > 0:  # maximize log p(x) p_x_weight是啥
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
                L_fp = -fp
                L_fq = fq
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
                L += args.p_x_weight * l_p_x + args.l2 * ((fp_all**2).mean() + (fq_all**2).mean()) ##这里的意思应该是 p(x)的权重，p(x)和p(y|x)设置不同的权重有利于正则化?

            
            if args.p_y_given_x_weight > 0:  # maximize log p(y | x) 如果是uncond这里也会算cross_entropy吗
                if not args.uncond:
                    logits = f.classify(x_lab) 
                    l_p_y_given_x = nn.CrossEntropyLoss()(logits, y_lab)
                    if cur_iter % args.print_every == 0: ##args.print_every=100
                        acc = (logits.max(1)[1] == y_lab).float().mean()
                        writer.add_scalar(tag="iter/loss_classify", scalar_value=l_p_y_given_x.item(),
                            global_step=global_step)
                        writer.add_scalar(tag="iter/acc", scalar_value=acc.item(),
                            global_step=global_step)
                        print('P(y|x) {}:{:>d} loss={:>14.9f}, acc={:>14.9f}'.format(epoch,
                                                                                    cur_iter,
                                                                                    l_p_y_given_x.item(),
                                                                                    acc.item()))
                    L += args.p_y_given_x_weight * l_p_y_given_x

            if args.p_x_y_weight > 0:  
                # maximize log p(x, y) 这里的意思应该是直接把(x,y)当作一个整体，去最大化这个整体的似然
                # f函数会返回 return t.gather(logits, 1, y[:, None]) #返回(x,y)的概率，y是一个数组(代表我有用到的label)
                # 应该跟上面分成 log p(y|x) + log p(x)是等价的，还有待考证
                assert not args.uncond, "this objective can only be trained for class-conditional EBM DUUUUUUUUHHHH!!!"
                x_q_lab = sample_q(f, replay_buffer, num_gred_des, y=y_lab)
                fp_all = f(x_lab, y_lab)
                fq_all = f(x_q_lab, y_lab)
                fp, fq = fp_all.mean(), fq_all.mean()
                l_p_x_y = -(fp - fq)
                if cur_iter % args.print_every == 0: ##args.print_every=100
                    print('P(x, y) | {}:{:>d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f} d={:>14.9f}'.format(epoch, i, fp, fq,
                                                                                                      fp - fq))

                L += args.p_x_y_weight * l_p_x_y + args.l2 * ((fp_all**2).mean() + (fq_all**2).mean()) 
            #if cur_iter % args.print_every == 0: ##args.print_every=100
            #    writer.add_scalar(tag="iter/loss_generative", scalar_value=L,
            #              global_step=global_step)

            writer.add_scalar(tag="iter/loss_generative", scalar_value=L,
                          global_step=global_step)
            if args.uncond:
                writer.add_scalars("iter/loss_generative_partition", {'L_fp': L_fp, 
                                    'L_fq': L_fq},
                                    global_step=global_step)
            else:
                writer.add_scalars("iter/loss_generative_partition", {'L_fp': L_fp, 
                                    'L_fq': L_fq,
                                    'L_CrossEntropy': l_p_y_given_x},
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
                    writer.add_images('x_q_image_epoch{}_iter{:>06d}'.format(epoch,cur_iter), x_q, global_step)
                if args.plot_cond:  # generate class-conditional samples
                    y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                    x_q_y = sample_q(f, replay_buffer, num_gred_des, y=y)
                    E_q = -f(x_q_y)
                    plot('{}/x_q_y{}_{:>06d}.png'.format(args.save_dir, epoch, i), x_q_y)
                    writer.add_images('x_q_y_image_epoch{}_iter{:>06d}'.format(epoch,cur_iter), x_q_y, global_step)

        if epoch % args.ckpt_every == 0: # 每10个epoch搞一个checkpoint
            checkpoint(f, replay_buffer, num_gred_des, f'ckpt_{epoch}.pt', args, device)

        if epoch % args.eval_every == 0 and (args.p_y_given_x_weight > 0 or args.p_x_y_weight > 0): #每一个epoch evaluation一次
            f.eval() ## Sets the module in evaluation mode.
            with t.no_grad():
                # validation set
                correct_valid, loss_valid = eval_classification(f, dload_valid, device)
                print("Epoch {} Global_step {} : Valid Loss {}, Valid Acc {}".format(epoch, global_step, loss_valid, correct_valid))
                if correct_valid > best_valid_acc:
                    best_valid_acc = correct_valid
                    print("Best Valid!: {}".format(correct_valid))
                    checkpoint(f, replay_buffer, num_gred_des, "best_valid_ckpt.pt", args, device)
                # test set
                correct_test, loss_test = eval_classification(f, dload_test, device)
                print("Epoch {} Global_step {}: Test Loss {}, Test Acc {}".format(epoch, global_step, loss_test, correct_test))
                writer.add_scalars("epoch/loss_classify", {'valid': loss_valid, 
                                    'test': loss_test},
                                    epoch)
                writer.add_scalars("epoch/acc_classify", {'valid': correct_valid, 
                                    'test': correct_test},
                                    epoch)
            if args.plot_uncond: # generate class-conditional samples生成不带标签的样本
                if args.class_cond_p_x_sample:
                    assert not args.uncond, "can only draw class-conditional samples if EBM is class-cond"
                    y_q = t.randint(0, args.n_classes, (args.batch_size,)).to(device)
                    x_q = sample_q(f, replay_buffer, num_gred_des, y=y_q)
                else:
                    x_q = sample_q(f, replay_buffer, num_gred_des,)
                writer.add_images('x_q_image_epoch{}'.format(epoch), x_q, epoch)
                fid , isc = cond_fid_is(f, x_q, args, device, args.ratio)
                writer.add_scalars("epoch/fid_is", {'FID': fid, 
                                    'ISC': isc},
                                    epoch)
            if args.plot_cond:  # generate class-conditional samples
                y = t.arange(0, args.n_classes)[None].repeat(args.n_classes, 1).transpose(1, 0).contiguous().view(-1).to(device)
                x_q_y = sample_q(f, replay_buffer, num_gred_des, y=y)
                writer.add_images('x_q_y_image_epoch{}'.format(epoch), x_q_y, epoch)
                fid , isc = cond_fid_is(f, x_q_y, args, device, args.ratio)
                writer.add_scalars("epoch/fid_is", {'FID': fid, 
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
    parser.add_argument("--p_y_given_x_weight", type=float, default=1.)
    parser.add_argument("--p_x_y_weight", type=float, default=0.)
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