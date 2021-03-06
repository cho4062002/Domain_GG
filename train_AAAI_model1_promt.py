# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import json
import os
import pickle


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.TransformersDataset import ImageDataset

import timm

assert timm.__version__ == "0.4.12"  # version check "0.3.2"
import timm.optim.optim_factory as optim_factory
from timm.utils import accuracy
from model_glance_AAAI_1 import VisionTransformer_glance
from model_gaze_AAAI_1 import CAT




def get_args_parser():
    parser = argparse.ArgumentParser('DG pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default= '/mnt/md0/dataset/imagenet/ILSVRC/Data/CLS-LOC/', type=str, # '/HDD/dataset/imagenet/ILSVRC/Data/CLS-LOC
                        help='dataset path')

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def dfs_freeze(model):
    for child in model.named_children():
        for param in child.parameters():
            print(param)
            param.requires_grad = False
            print(param)
        dfs_freeze(child)

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print(device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(seed)
    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    model_glance = VisionTransformer_glance()
    model_gaze = CAT()
    criterion = nn.CrossEntropyLoss()
    model_glance.to(device)
    model_gaze.to(device)

    print("accumulate grad iterations: %d" % args.accum_iter)

    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_glance, args.weight_decay)
    param_groups1 = optim_factory.add_weight_decay(model_gaze, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer1 = torch.optim.AdamW(param_groups1, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    #print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    lambda_ = 0.3
    lambda_2 = 0.7
    epoch_acc1_=[]
    epoch_acc5_=[]
    loss_save_=[]
    for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0.0
        loss = 0.0
        loss_save = 0.0
        acc1 = 0.0
        acc5 = 0.0 
        acc1_save = 0.0
        acc5_save = 0.0
        epoch_total_acc1 =0.0
        epoch_total_acc5 =0.0
        for i, data in enumerate(data_loader_train):
            inputs , labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            outputs_glance = model_glance(inputs)
            loss_glance = criterion(outputs_glance, labels)
            loss_glance.backward()
            optimizer.step()
            #model_freeze_glance.to(device)
            outputs_gaze = model_gaze(inputs)
            outputs_total = model_glance(outputs_gaze, True)
            loss_total = criterion(outputs_total, labels)



            loss_total.backward()
            optimizer.step()
            optimizer1.step()
            acc1_total, acc5_total = accuracy(outputs_total, labels, topk=(1, 5))
        # print statistics
            loss += loss_total
            acc1 += acc1_total.item()
            acc5 += acc5_total.item()
            running_loss += loss_total
            epoch_total_acc1 += acc1_total.item()
            epoch_total_acc5 += acc5_total.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
                print('epoch: %d , acc1: %.3f acc5: %.3f' % (epoch + 1,epoch_total_acc1 / 50, epoch_total_acc5 / 50))
                running_loss = 0.0
                epoch_total_acc1 = 0.0
                epoch_total_acc5 = 0.0
        loss_save = loss/i
        acc1_save = acc1/i
        acc5_save = acc5/i
        loss_save_.append(loss_save)
        epoch_acc1_.append(acc1_save)
        epoch_acc5_.append(acc5_save)

        batch_time = time.time() - start_time
        print(batch_time)
        print('-----------------------------------------')

        #running_loss = 0.0
        #epoch_total_acc1 =0.0
        #epoch_total_acc5 =0.0
        #for i, data in enumerate(data_loader_train):
        #    inputs , labels = data
        #    inputs, labels = inputs.to(device), labels.to(device)
        #    optimizer.zero_grad()
        #    optimizer1.zero_grad()
        #    outputs_glance = model_glance(inputs)
        #    outputs_gaze = model_gaze(inputs)
        #    outputs_total = model_glance(outputs_gaze)
        #    loss_total = criterion(outputs_total, labels)
        #    loss_total.backward()
        #    optimizer.step()
        #    optimizer1.step()
        #    acc1_total, acc5_total = accuracy(outputs_total, labels, topk=(1, 5))
        # print statistics
        #    running_loss += loss_total
        #    epoch_total_acc1 += acc1_total.item()
        #    epoch_total_acc5 += acc5_total.item()
        #    
        #    if i % 50 == 49:    # print every 2000 mini-batches

        #        print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
        #        print('epoch: %d , acc1: %.3f acc5: %.3f' % (epoch + 1,epoch_total_acc1 / 50, epoch_total_acc5 / 50))
        #        running_loss = 0.0
        #        epoch_total_acc1 = 0.0
        #        epoch_total_acc5 = 0.0
        #batch_time = time.time() - start_time
        #print(batch_time)
        #print('-----------------------------------------')
       

              

        if (epoch % 1 == 0 or epoch + 1 == args.epochs):
            PATH = '/mnt/md0/yunsung/AAAI_1'

            # torch.save(model_glance, PATH_glance + 'model_glance.pt')  # ?????? ?????? ??????
            # torch.save(model_glance.state_dict(), PATH_glance +str(epoch)+ '_epoch_model_glance_state_dict.pt')  # ?????? ????????? state_dict ??????
            torch.save({
                        'model_glance': model_glance.state_dict(),
                        'model_gaze': model_gaze.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'optimizer1': optimizer1.state_dict()
                        }, PATH +'all.tar')
            # PATH_gaze = '/home/yunsung/glance_gaze/AAAI_version1/train_model_gaze_AAAI_version1/'
            # torch.save(model_gaze, PATH_gaze + 'model_gaze.pt')  # ?????? ?????? ??????
            # torch.save(model_gaze.state_dict(), PATH_gaze +str(epoch) + '_epoch_model_gaze_state_dict.pt')  # ?????? ????????? state_dict ??????
            # torch.save({
            #             'model': model_gaze.state_dict(),
            #             'optimizer': optimizer1.state_dict()
            #             }, PATH_gaze + 'all.tar')
            with open("/mnt/md0/yunsung/AAAI_1/ouput.pkl", 'wb') as f:
                pickle.dump({'acc1': epoch_acc1_, 'acc5': epoch_acc5_,
                            'loss': loss_save_ }, f)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    main(args)
