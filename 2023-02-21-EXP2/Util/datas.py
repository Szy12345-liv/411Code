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

seed = 1
im_sz = 32
n_ch = 3


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

def cycle(loader):
    while True:
        for data in loader:
            yield data

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