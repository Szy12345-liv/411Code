import torch as t
import torch.nn as nn
from models import wideresnet
import models
im_sz = 32
n_ch = 3

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