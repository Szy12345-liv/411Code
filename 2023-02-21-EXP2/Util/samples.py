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
from models.ebm_models import init_random

seed = 1
im_sz = 32
n_ch = 3

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
    def random_sampler_branch(f, bs, n_steps=args.n_steps, y=None):
        random_samples = t.FloatTensor(bs, n_ch, im_sz, im_sz).uniform_(-1, 1)
        random_samples = random_samples.to(device)
        x_k = t.autograd.Variable(random_samples, requires_grad=True)
        for k in range(n_steps):
            f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
            x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k) ##这里sgld的lr和std不应该是根号关系吗？为何可以自己调整噪声
        f.train() ##这是什么意思 Sets the module in training mode.
        final_samples = x_k.detach() ##采出来的样本不用求导了
        return final_samples
    def random_sampler(f, bs, n_steps=args.n_steps, y=None):
        n_it = bs // 64
        buffer = []
        for i in tqdm(range(n_it)):
            x_q = random_sampler_branch(f, 64, n_steps, y)
            buffer.append(x_q)
        buffer = t.cat(buffer, 0)
        return buffer
    return sample_q, track_q, random_sampler