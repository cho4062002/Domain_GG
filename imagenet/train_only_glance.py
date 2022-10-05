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
import numpy as np
import os
import time
from pathlib import Path
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


import json
import os
import pickle


import torch
# import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.TransformersDataset import ImageDataset

import timm

assert timm.__version__ == "0.4.12"  # version check "0.3.2"
import timm.optim.optim_factory as optim_factory
from timm.utils import accuracy
from model_glance import glance
from model_gaze import CAT
from model_classification_glance import Classification




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
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default= '/mnt/storage1/dataset/imagenet/ILSVRC/Data/CLS-LOC', type=str, # '/HDD/dataset/imagenet/ILSVRC/Data/CLS-LOC
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


    return parser


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
    # cudnn.benchmark = True

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

    # define the model
    # model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model_glance = glance()
    model_gaze = CAT()
    model_classification = Classification()
    criterion = nn.CrossEntropyLoss()
    model_glance.to(device)
    model_gaze.to(device)
    model_classification.to(device)

    print("accumulate grad iterations: %d" % args.accum_iter)

    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_glance, args.weight_decay)
    param_groups1 = optim_factory.add_weight_decay(model_gaze, args.weight_decay)
    param_groups2 = optim_factory.add_weight_decay(model_classification, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer1 = torch.optim.AdamW(param_groups1, lr=args.lr, betas=(0.9, 0.95))
    optimizer2 = torch.optim.AdamW(param_groups2, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)

    #print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    lambda_ = 0.3
    lambda_2 = 0.7
    epoch_acc1_=[]
    epoch_acc5_=[]
    loss_save_=0.0
    loss_=[]
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
        epoch_total_acc1 =0.0
        epoch_total_acc5 =0.0
        for i, data in enumerate(tqdm(data_loader_train)):
            inputs , labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer2.zero_grad()
            outputs_glance = model_glance(inputs)
            outputs_total = outputs_glance
            outputs_total = model_classification(outputs_glance)
            loss_total = criterion(outputs_total, labels)
            #loss_glance.backward()
            #loss_gaze.backward()
            loss_total.backward()
            optimizer.step()
            optimizer2.step()
            acc1_total, acc5_total = accuracy(outputs_total, labels, topk=(1, 5))
        # print statistics
            running_loss += loss_total
            epoch_total_acc1 += acc1_total.item()
            epoch_total_acc5 += acc5_total.item()
            
        loss_save = running_loss/i
        acc1_save = epoch_total_acc1/i
        acc5_save = epoch_total_acc5/i
        loss_.append(loss_save)
        epoch_acc1_.append(acc1_save)
        epoch_acc5_.append(acc5_save)
        batch_time = time.time() - start_time
        print('epoch: %d , epoch_loss: %.3f epoch_acc1: %.3f epoch_acc5: %.3f' % (epoch + 1, loss_save, acc1_save, acc5_save))
        print(batch_time)
        
       

              

        if (epoch % 1 == 0 or epoch + 1 == args.epochs):
            PATH_glance = '/mnt/storage1/yunsung/imagenet_result/only_glance/glance/' 
            torch.save(model_glance.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_model_state_dict.pt')
            torch.save(optimizer.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_opti_state_dict.pt') 
            torch.save({
                        'model_glance': model_glance.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, PATH_glance + 'all.tar')
            PATH_classification = '/mnt/storage1/yunsung/imagenet_result/only_glance/classification/'
            torch.save(model_classification.state_dict(), PATH_classification +str(epoch) + '_epoch_classification_model_state_dict.pt')  
            torch.save(optimizer2.state_dict(), PATH_classification +str(epoch)+ '_epoch_classification_opti_state_dict.pt')  
            torch.save({
                        'model_classification': model_classification.state_dict(),
                        'optimizer2': optimizer2.state_dict()
                        }, PATH_classification + 'all.tar')
            with open("/mnt/storage1/yunsung/imagenet_result/only_glance/result/ouput.pkl", 'wb') as f:
                    pickle.dump({'acc1': epoch_acc1_, 'acc5': epoch_acc5_,
                                'loss': loss_ }, f)
        with open("/mnt/storage1/yunsung/imagenet_result/only_glance/result/ouput_.pkl", 'wb') as f:
                    pickle.dump({'acc1': epoch_acc1_, 'acc5': epoch_acc5_,
                                'loss': loss_ }, f)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
