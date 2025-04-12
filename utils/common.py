from __future__ import absolute_import
import datetime
import shutil
from pathlib import Path
import os
import random
import numpy as np
import torch
import logging
import torch.nn as nn
# from torch.optim.lr_scheduler import LRScheduler

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
     
def ensure_path(directory):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
 
def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])  
    else:
        return
    os.mkdir(path)

'''record configurations'''
class record_config():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)

        # config_dir = self.job_dir / 'config.txt'
        if args.prun_att:
            config_dir = self.job_dir / 'atten_config.txt'
        else:
            config_dir = self.job_dir / 'config.txt'
        if args.resume != None:
            with open(config_dir, 'a') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')
        else:
            with open(config_dir, 'w') as f:
                f.write(now + '\n\n')
                for arg in vars(args):
                    f.write('{}: {}\n'.format(arg, getattr(args, arg)))
                f.write('\n')

class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.ckpt_dir = self.job_dir / 'checkpoint'
        self.run_dir = self.job_dir / 'run'

        def _make_dir(path):
            if not os.path.exists(path):
                os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.ckpt_dir)
        _make_dir(self.run_dir)

        record_config(args)

    def save_model(self, state, epoch, is_best,bestpath):
        save_path = f'{self.ckpt_dir}/model_last.pt'
        torch.save(state, save_path)
        best_path=None
        if is_best:
            if os.path.exists(bestpath):
                os.remove(bestpath)
            best_path = f'{self.ckpt_dir}/ {epoch}_{state["best_acc"].cpu().item()}_model_best.pt'
            shutil.copyfile(save_path, best_path)

        return best_path

    def save_class_model_v1(self, state):
        save_path = f'{self.ckpt_dir}/class_model_last.pt'
        torch.save(state, save_path)

    def save_class_model(self, state,epoch, is_best,bestpath):
        save_path = f'{self.ckpt_dir}/class_model_last.pt'
        torch.save(state, save_path)
        best_path = None
        if is_best:
            if os.path.exists(bestpath):
                os.remove(bestpath)
            best_path = f'{self.ckpt_dir}/ {epoch}_{state["best_acc"]}_class_model_best.pt'
            shutil.copyfile(save_path, best_path)

        return best_path


    def save_pretrain_model(self, state):
        save_path = f'{self.ckpt_dir}/pretrain_model.pt'
        # print('=> Saving model to {}'.fo_rmat(save_path))
        torch.save(state, save_path)

def get_logger(file_path):

    if not os.path.exists(file_path):
        os.open(file_path,os.O_CREAT)

    logger = logging.getLogger('gal')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger

"""Computes the precision@k for the specified values of k"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))
        return res

def direct_project(weight, indices):
    #print(weight.size())

    A = torch.randn(weight.size(0), len(indices), weight.size(2), weight.size(3))
    #print(A.size())
    for i, indice in enumerate(indices):

        A[:, i, :, :] = weight[:, indice, :, :]

    return A


# class WarmUpLR(LRScheduler):
#     """warmup_training learning rate scheduler
#     Args:
#         optimizer: optimzier(e.g. SGD)
#         total_iters: totoal_iters of warmup phase
#     """
#     def __init__(self, optimizer, total_iters, last_epoch=-1-0.5):
#
#         self.total_iters = total_iters
#         super().__init__(optimizer, last_epoch)
#
#     def get_lr(self):
#         """we will use the first m batches, and set the learning
#         rate to base_lr * m / total_iters
#         """
#         return [base_lr * self.last_epoch / (self.total_iters + 1e-8_0.4_o) for base_lr in self.base_lrs]


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse():
    import argparse

    parser = argparse.ArgumentParser(description='White-Box')

    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        default=[0],
        help='Select gpu_id to use. default:[0]',
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='Select dataset to train. default:cifar10',
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default='/data/cifar10/',
        help='The dictionary where the input is stored. default:/data/cifar10/',
    )

    parser.add_argument(
        '--job_dir',
        type=str,
        default='experiments/',
        help='The directory where the summaries will be stored. default:./experiments'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Load the model from the specified checkpoint.'
    )

    ## Training
    parser.add_argument(
        '--arch',
        type=str,
        default='resnet',
        help='Architecture of model. default:resnet'
    )

    parser.add_argument(
        '--cfg',
        type=str,
        default='resnet50',
        help='Detail architecuture of model. default:resnet50'
    )

    parser.add_argument(
        '--num_epochs',
        type=int,
        default=300,
        help='The number of epoch to train. default:300'
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=256,
        help='Batch size for training. default:256'
    )

    parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=100,
        help='Batch size for validation. default:100'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer. default:0.9'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-2,
        help='Learning rate for train. default:1e-2'
    )

    parser.add_argument(
        '--lr_type',
        default='step',
        type=str,
        help='lr scheduler (step/exp/cos/step3/fixed)'
    )

    parser.add_argument(
        '--criterion',
        default='Softmax',
        type=str,
        help='Loss func (Softmax)'
    )

    parser.add_argument(
        '--lr_decay_step',
        type=int,
        nargs='+',
        default=[50, 100],
        help='the iterval of learn rate. default:50, 100'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-3,
        help='The weight decay of loss. default:5e-3'
    )

    parser.add_argument(
        '--pruning_rate',
        type=float,
        default=0.5,
        help='Target Pruning Rate. default:0.5'
    )

    parser.add_argument(
        '--classtrain_epochs',
        type=int,
        default=30,
        help='Train_class_epochs'
    )

    parser.add_argument(
        '--sparse_lambda',
        type=float,
        default=0.0001,
        help='Sparse_lambda. default:0.00001'
    )

    parser.add_argument(
        '--min_preserve',
        type=float,
        default=0.3,
        help='Minimum preserve percentage of each layer. default:0.3'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='input to open debug state'
    )

    parser.add_argument(
        '--warm',
        type=int,
        default=0,
        help='warm'
    )

    parser.add_argument(
        '--scheduler',
        type=bool,
        default=True,
        help='scheduler'
    )
    parser.add_argument(
        '--ccm',
        type=float,
        default=0.01,
        help='scheduler'
    )

    parser.add_argument(
        '--class_resume',
        type=bool,
        default=True,
        help='class_resume'
    )

    parser.add_argument(
        '--prun_xinxi',
        type=bool,
        default=False,
        help='prun_xinxi'
    )
    parser.add_argument(
        '--prun_threshold',
        type=float,
        default=0.9,
        help='prun_threshold'
    )

    parser.add_argument(
        '--prun_att',
        type=bool,
        default=False,
        help='prun_att'
    )
    parser.add_argument(
        '--train_resume',
        type=bool,
        default=False,
        help='train_resume'
    )

    parser.add_argument(
        '--prun_att_model',
        type=str,
        default='CBAMBlock',
        help='prun_att_model'
    )
    parser.add_argument(
        '--prun_att_channel',
        type=int,
        default=50,
        help='prun_att_channel'
    )
    parser.add_argument(
        '--class_ccm',
        type=bool,
        default=True,
        help='class_ccm'
    )
    parser.add_argument(
        '--class_attenion',
        type=bool,
        default=True,
        help='class_attenion'
    )
    parser.add_argument(
        '--prun_ccm_batchweight',
        type=bool,
        default=False,
        help='prun_ccm_batchweight'
    )

    parser.add_argument(
        '--train_noise',
        type=bool,
        default=True,
        help='train_noise'
    )

    parser.add_argument(
        '--train_resume_checkpoint',
        type=str,
        default="",
        help='train_resume_checkpoint'
    )

    parser.add_argument(
        '--train_noise_temp',
        type=str,
        default='',
        help='train_noise_temp'
    )
    parser.add_argument(
        '--train_noise_ccm',
        type=float,
        default='0.1',
        help='train_noise_ccm'
    )



    args = parser.parse_args()
    return args

def heatmap(index,data):
    # 创建一个随机数据的DataFrame
    np.random.seed(0)
    df = pd.DataFrame(data)
    f, ax = plt.subplots(figsize=(14, 10))
    # 创建热图，指定颜色映射（形状）
    sns.heatmap(df, vmax=1, vmin=0)
    ax.invert_yaxis()
    f.savefig("layer{}.jpg".format(index))
    # 显示图形
    f.show()


def savenpy(layerIndex,data,batch):
    path="E:\work\code_rebuild\WhiteCRC\experiment\cifar10\\resnet56\\npy\\{}".format(batch)
    if os.path.exists(path)==False:
        os.mkdir(path)
    path=os.path.join(path,str(layerIndex)+'.npy')
    np.save(path, data)


def decrease_with_x(x):
    return 10 / (1 + math.exp(x / 70 - 2))
