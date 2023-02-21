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

def write_images(tag ,images ,step ,writer):
    images = t.clamp(images, -1, 1)
    images = images * 0.5 + 0.5
    writer.add_images(tag ,images ,step)