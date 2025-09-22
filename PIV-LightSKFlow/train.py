from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import sys
import cv2
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from models import *
from utils import flow_viz
import datasets
# 移除evaluate模块的导入
from torch.cuda.amp import GradScaler

MAX_FLOW = 400

def convert_flow_to_image(image1, flow):
    flow = flow.permute(1, 2, 0).cpu().numpy()
    flow_image = flow_viz.flow_to_image(flow)
    flow_image = cv2.resize(flow_image, (image1.shape[3], image1.shape[2]))
    return flow_image

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_args(args):
    print('----------- args ----------')
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    print('---------------------------')

def sequence_loss(flow_preds, flow_gt, valid, gamma):
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    flow_error = flow_preds[-1] - flow_gt
    squared_error = torch.sum(flow_error**2, dim=1)
    valid_mask = valid[:, 0]
    valid_squared_error = squared_error[valid_mask]
    mse = valid_squared_error.mean()
    rmse = torch.sqrt(mse)

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'rmse': rmse.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics

def fetch_optimizer(args, model):
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                            pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler

class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_rmse_list = []
        self.train_steps_list = []
        # 移除验证相关的记录
        self.terminal = sys.stdout
        self.log = open(self.args.output+'/log.txt', 'a')

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int32)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        current_time = time.strftime("%Y-%m-%d %H:%M:%S") + "  "
        print(current_time + training_str + metrics_str + time_left_hms)
        sys.stdout.flush()

        self.train_rmse_list.append(np.mean(self.running_loss_dict['rmse']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []
            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main(args):
    model = eval(args.model_name)(args).cuda()
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=True)

    model.cuda()
    model.train()
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler, args)
    sys.stdout = logger
    print_args(args)
    print(f"Parameter Count: {count_parameters(model)}")

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, scaler, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)  # 仅保留训练结果绘图
            break

    PATH = args.output+f'/{args.name}.pth'
    torch.save(model.state_dict(), PATH)
    return PATH

def train(model, train_loader, optimizer, scheduler, logger, scaler, args):
    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()
        image1, image2, flow, valid = [x.cuda() for x in data_blob]

        optimizer.zero_grad()
        flow_pred = model(image1, image2)

        loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        logger.push(metrics)
 
         # 检查当前步数是否为100000的倍数，并保存模型
        current_step = logger.total_steps
        if current_step % 100000 == 0 and current_step != 0:
            save_path = os.path.join(args.output, f'{args.name}_step_{current_step}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"\nSaved model checkpoint at step {current_step} to {save_path}")
            
            # 生成当前步数的对数刻度图
            plt.figure()
            plt.plot(logger.train_steps_list, logger.train_rmse_list)
            plt.yscale('log')
            plt.xlabel('Training Steps')
            plt.ylabel('RMSE (log scale)')
            plt.title(f'Training RMSE at Step {current_step}')
            plot_filename = os.path.join(args.output, f'train_rmse_log_step_{current_step}.png')
            plt.savefig(plot_filename, bbox_inches='tight')
            plt.close()
            print(f"Saved log-scale plot at step {current_step} to {plot_filename}")
                   
        if logger.total_steps >= args.num_steps:
            break

def plot_train(logger, args):
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_rmse_list)
    plt.xlabel('x_steps')
    plt.ylabel('RMSE')
    plt.title('Running training error (RMSE)')
    plt.savefig(args.output+"/train_rmse.png", bbox_inches='tight')
    plt.close()

    # 新增对数刻度图
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_rmse_list)
    plt.yscale('log')  # 设置纵轴为对数刻度
    plt.xlabel('x_steps')
    plt.ylabel('RMSE (log scale)')
    plt.title('Running training error (RMSE) - Log Scale')
    plt.savefig(args.output+"/train_rmse_log.png", bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='bla', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    # 移除validation参数
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--output', type=str, default='checkpoints', help='output directory')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, nargs='+', default=[256, 256])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--iters', type=int, default=12)
    # 移除val_freq参数
    parser.add_argument('--print_freq', type=int, default=100)

    parser.add_argument('--mixed_precision', default=False, action='store_true')
    parser.add_argument('--model_name', default='RAFTGMA', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true')
    parser.add_argument('--position_and_content', default=False, action='store_true')
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--piv_root', type=str, default=None)
    parser.add_argument('--k_conv', type=int, nargs='+', default=[1, 15])
    parser.add_argument('--UpdateBlock', type=str, default='SKUpdateBlock6')
    parser.add_argument('--PCUpdater_conv', type=int, nargs='+', default=[1, 7])

    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    main(args)
