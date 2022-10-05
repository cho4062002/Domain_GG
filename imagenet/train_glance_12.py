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
import pandas as pd
import json
import os
import pickle

import torch
import torchvision.transforms as transforms

import torchvision.datasets as datasets
from data.TransformersDataset import ImageDataset
# from pacs import PACS


import timm
assert timm.__version__ == "0.4.12"  # version check "0.3.2"
import timm.optim.optim_factory as optim_factory
from model_glance_12 import glance
from model_gaze import CAT
from model_classification import Classification
import torch
from timm.utils import accuracy

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
    # train_dataset = PACS(root='C:/Users/cho40/pacs/', 
    #                                domain= 'art_painting',
    #                                phase='train')

    

    # data_loader_train = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    ###############################################################################################
    model_glance = glance()
    model_gaze = CAT()
    model_classification = Classification()
    model_glance.to(device)
    model_gaze.to(device)
    model_classification.to(device)

    criterion = nn.CrossEntropyLoss()

    param_groups = optim_factory.add_weight_decay(model_glance, args.weight_decay)
    param_groups1 = optim_factory.add_weight_decay(model_gaze, args.weight_decay)
    param_groups2 = optim_factory.add_weight_decay(model_classification, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    optimizer1 = torch.optim.AdamW(param_groups1, lr=args.lr, betas=(0.9, 0.95))
    optimizer2 = torch.optim.AdamW(param_groups2, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    ###############################################################################################
    #train
    def Train(data_loader_train, criterion, device, optimizer, optimizer1, optimizer2, model_glance, model_gaze, model_classification):
        model_glance.train()
        model_gaze.train()
        model_classification.train()

        ### start_time = time.time()

        epoch_acc_=[]
        epoch_acc5_=[]

        loss_=[]
        #for epoch in range(args.start_epoch, args.epochs):
        running_loss = 0.0
        loss_save = 0.0
        acc1_save = 0.0
        acc5_save = 0.0
        epoch_total_acc1 =0.0
        epoch_total_acc5 =0.0
        epoch_total_acc1 =0.0
        epoch_total_acc5 =0.0
        for i, data in enumerate(tqdm(data_loader_train)):
            inputs , labels = data
            # labels  = torch.stack(labels, dim=0)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            outputs_glance = model_glance(inputs)
            outputs_gaze = model_gaze(inputs)
            outputs_total = torch.cat([outputs_glance, outputs_gaze], dim = 1)
            outputs_total = model_classification(outputs_total)
            # outputs_total = model_classification(outputs_glance)
            loss_total = criterion(outputs_total, labels)
            loss_total.backward()
            optimizer.step()
            optimizer1.step()
            optimizer2.step()
            
            # print(labels)
            # print(outputs_total)
            acc1_total, acc5_total = accuracy(outputs_total, labels, topk=(1, 5))
            # # print statistics

            running_loss += loss_total
            epoch_total_acc1 += acc1_total.item()
            epoch_total_acc5 += acc5_total.item()
            # print(outputs_total[1])
            # print(labels[1])
        
        
        loss_save = running_loss/i
        acc1_save = epoch_total_acc1/i
        acc5_save = epoch_total_acc5/i
        loss_.append(loss_save)
        ### epoch_acc_.append(acc_save)
        ### batch_time = time.time() - start_time
        ### print('epoch: %d , epoch_loss: %.3f epoch_acc: %.3f ' % (epoch + 1, loss_save, acc_save))
        ### print(batch_time)
        return loss_save, acc1_save, acc5_save

    def Evaluation(data_loader_val, criterion, device, model_glance, model_gaze, model_classification):  
        ### start_time = time.time()  
        model_glance.eval()
        model_gaze.eval()
        model_classification.eval()   

        running_loss_val =0
        loss_save_ = 0
        acc1_save_ = 0.0
        acc5_save_ = 0.0
        epoch_total_acc1_ =0.0
        epoch_total_acc5_ =0.0
        with torch.no_grad():

            for i, data in enumerate(tqdm(data_loader_val)):
                inputs , labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs_glance_ = model_glance(inputs)
                outputs_gaze_ = model_gaze(inputs)
                outputs_total_ = torch.cat([outputs_glance_, outputs_gaze_], dim = 1)
                outputs_total_ = model_classification(outputs_total_)
                # outputs_total_ = model_classification(outputs_glance_)
                loss_total_ = criterion(outputs_total_, labels)

                
                # print(labels)
                # print(outputs_total)
                acc1_total_, acc5_total_ = accuracy(outputs_total_, labels, topk=(1, 5))
            # # print statistics


                running_loss_val += loss_total_
                epoch_total_acc1_ += acc1_total_.item()
                epoch_total_acc5_ += acc5_total_.item()

                # print(outputs_total[1])
                # print(labels[1])
                # acc_ = (outputs_total_ == labels).float().mean()
                # epoch_total_acc_ += acc_
            loss_save_ = running_loss_val/i
            acc1_save_ = epoch_total_acc1_/i
            acc5_save_ = epoch_total_acc5_/i
            #loss_.append(loss_save)
            #epoch_acc_.append(acc_save)
            ### batch_time = time.time() - start_time
            ### print('epoch: %d , epoch_loss_val: %.3f epoch_acc_val: %.3f ' % (epoch + 1, loss_save_, acc_save_))
            ### print(batch_time)
        return loss_save_, acc1_save_, acc5_save_

    best_valid_loss = float('inf')
    train_losses = []
    train_acc1 = []
    train_acc5 = []
    valid_losses = []
    valid_acc1 = []
    valid_acc5 = []
    for epoch in range(args.start_epoch, args.epochs):
        
        start_time = time.time()
        train_loss, acc_train_acc1_save, acc_train_acc5_save = Train(data_loader_train, criterion, device, optimizer, optimizer1, optimizer2, model_glance, model_gaze, model_classification)
        valid_loss, valid_acc1_save, valid_acc5_save = Evaluation(data_loader_val, criterion, device, model_glance, model_gaze, model_classification)
        train_losses.append(train_loss)
        train_acc1.append(acc_train_acc1_save)
        train_acc5.append(acc_train_acc5_save)
        valid_losses.append(valid_loss)
        valid_acc1.append(valid_acc1_save)
        valid_acc5.append(valid_acc5_save)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
            PATH_glance = '/mnt/storage1/yunsung/imagenet_result/no_pre_glance_12/glance/' 
            torch.save(model_glance.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_model_state_dict.pt')
            torch.save(optimizer.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_opti_state_dict.pt') 
    #         
            PATH_gaze = '/mnt/storage1/yunsung/imagenet_result/no_pre_glance_12/gaze/'  #/mnt/md0/yunsung/AAAI_3/train_model_gaze_AAAI_version3
            torch.save(model_gaze.state_dict(), PATH_gaze +str(epoch) + '_epoch_gaze_model_state_dict.pt')  
            torch.save(optimizer1.state_dict(), PATH_gaze +str(epoch)+ '_epoch_gaze_opti_state_dict.pt')  
    #         
            PATH_classification = '/mnt/storage1/yunsung/imagenet_result/no_pre_glance_12/classification/'
            torch.save(model_classification.state_dict(), PATH_classification +str(epoch) + '_epoch_classification_model_state_dict.pt')  
            torch.save(optimizer2.state_dict(), PATH_classification +str(epoch)+ '_epoch_classification_opti_state_dict.pt')  
    #         
            print(f'[Info] Model has been updated - epoch: {epoch}')

        print('epoch: %d , epoch_loss_train: %.3f epoch_acc1: %.3f epoch_acc5: %.3f ' % (epoch + 1, train_loss, acc_train_acc1_save, acc_train_acc5_save))
        
        print('epoch: %d , epoch_loss_val: %.3f epoch_acc1: %.3f epoch_acc5: %.3f ' % (epoch + 1, valid_loss, valid_acc1_save, valid_acc5_save))
        
        df = pd.DataFrame([train_losses,train_acc1,train_acc5,valid_losses,valid_acc1,valid_acc5]).transpose()
        df.columns=['train_losses','train_acc1','train_acc5','valid_losses','valid_acc1','valid_acc5']
        df.to_excel('/mnt/storage1/yunsung/imagenet_result/no_pre_glance_12/result/result.xlsx', index=False)
    df = pd.DataFrame([train_losses,train_acc1,train_acc5,valid_losses,valid_acc1,valid_acc5]).transpose()
    df.columns=['train_losses','train_acc1','train_acc5','valid_losses','valid_acc1','valid_acc5']
    df.to_excel('/mnt/storage1/yunsung/imagenet_result/no_pre_glance_12/result/result__.xlsx', index=False)
    #     if (epoch % 1 == 0 or epoch + 1 == args.epochs):
    #         PATH_glance = '/mnt/storage1/yunsung/grad/no_pretrain/glance/' 
    #         torch.save(model_glance.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_model_state_dict.pt')
    #         torch.save(optimizer.state_dict(), PATH_glance +str(epoch)+ '_epoch_glance_opti_state_dict.pt') 
    #         torch.save({
    #                     'model_glance': model_glance.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     }, PATH_glance + 'all.tar')
    #         PATH_gaze = '/mnt/storage1/yunsung/grad/no_pretrain/gaze/'  #/mnt/md0/yunsung/AAAI_3/train_model_gaze_AAAI_version3
    #         torch.save(model_gaze.state_dict(), PATH_gaze +str(epoch) + '_epoch_gaze_model_state_dict.pt')  
    #         torch.save(optimizer1.state_dict(), PATH_gaze +str(epoch)+ '_epoch_gaze_opti_state_dict.pt')  
    #         torch.save({
    #                     'model_gaze': model_gaze.state_dict(),
    #                     'optimizer1': optimizer1.state_dict()
    #                     }, PATH_gaze + 'all.tar')
    #         PATH_classification = '/mnt/storage1/yunsung/grad/no_pretrain/classification/'
    #         torch.save(model_classification.state_dict(), PATH_classification +str(epoch) + '_epoch_classification_model_state_dict.pt')  
    #         torch.save(optimizer2.state_dict(), PATH_classification +str(epoch)+ '_epoch_classification_opti_state_dict.pt')  
    #         torch.save({
    #                     'model_classification': model_classification.state_dict(),
    #                     'optimizer2': optimizer2.state_dict()
    #                     }, PATH_classification + 'all.tar')
    #         with open("/mnt/storage1/yunsung/grad/no_pretrain/result/ouput.pkl", 'wb') as f:
    #                 pickle.dump({'acc1': epoch_acc_, 'loss': loss_ }, f)
    #     with open("/mnt/storage1/yunsung/grad/no_pretrain/result/ouput__________.pkl", 'wb') as f:
    #                 pickle.dump({'acc': epoch_acc_, 'loss': loss_ }, f)
    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)