import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from configs.common_config import config as cfg
from models.backbone import resnet
from models.backbone import fpn


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class Net(nn.Module):
    def __init__(self,resnet_type,feature_dim_list):
        super().__init__()
        #resnet_type = cfg['resnet_type']
        #out_dim = cfg['data_label_dim']
        #feature_dim_list = [2048, 1024, 512, 256]
        #mix_tier_num = 4
        self.resnet = resnet.Net(resnet_type, cfg['input_dim'])
        self.fpn = fpn.Net(feature_dim_list)
        self.up = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)


        self.mix = conv1x1(256 * (len(feature_dim_list)-1), 256)
        self.up1 = nn.ConvTranspose2d(feature_dim_list[1], feature_dim_list[1], kernel_size=4, stride=2, padding=1)
        self.reduce1 = conv1x1(feature_dim_list[1], feature_dim_list[-1])
        self.up2 = nn.ConvTranspose2d(feature_dim_list[2], feature_dim_list[2], kernel_size=4, stride=2, padding=1)
        self.reduce2 = conv1x1(feature_dim_list[2], feature_dim_list[-1])
        self.up3 = nn.ConvTranspose2d(feature_dim_list[3], feature_dim_list[3], kernel_size=4, stride=2, padding=1)

        self.out_dim1 = conv1x1(256, 1)
        self.out_dim2 = conv1x1(256, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.act_prob = torch.sigmoid
        self.relu = F.relu

        self.BCE_loss = nn.BCELoss()

    def head(self, x):
        feature = x
        feature = self.up(feature)

        out_dim1 = self.out_dim1(feature)
        out_dim1 = self.act_prob(out_dim1)
        out_dim2 = self.out_dim2(feature)
        out_dim2 = self.act_prob(out_dim2)

        dim_list = [out_dim1, out_dim2]

        out = torch.cat(dim_list, dim=1)
        return out,feature

    def feature_mix(self, f_list):
        f_list[0] = self.up1(f_list[0])
        f_list[0] = self.up1(f_list[0])
        f_list[0] = self.up1(f_list[0])
        f_list[0] = self.reduce1(f_list[0])
        f_list[1] = self.up2(f_list[1])
        f_list[1] = self.up2(f_list[1])
        f_list[1] = self.reduce2(f_list[1])
        f_list[2] = self.up3(f_list[2])
        out = torch.cat(f_list, dim=1)
        out = self.mix(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

    def forward(self,x):
        # 下采样 提取特征
        feature, multi_scale = self.resnet(x)
        feature, f_list = self.fpn(feature, multi_scale, True)
        mix_feature = self.feature_mix(f_list)
        out,feature = self.head(mix_feature)
        return out,feature

    def loss_function(self, output, target,eps=1e-10):
        output = output.to(cfg['device'])
        target = target.to(cfg['device'])
        dim0_loss = self.BCE_loss(output[0][0],target[0][0])
        dim1_loss = self.BCE_loss(output[0][1],target[0][1])

        b_output = torch.where(output[0][0]-output[0][1]>0,output[0][0]/output[0][0],output[0][0]*0)
        b_target = torch.where(target[0][0]-target[0][1]>0,target[0][0]/target[0][0],target[0][0]*0)
        binary_loss = self.BCE_loss(b_output,b_target)

        total_loss = 1 * dim0_loss + 0.001 * binary_loss + 1 * dim1_loss
        return total_loss
