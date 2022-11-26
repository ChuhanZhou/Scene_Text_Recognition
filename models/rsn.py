import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from configs.common_config import config as cfg
from models.backbone import resnet
from models.backbone import fpn
from models import text_line_detection_net as ldn
from tool import process_data
from tool import nms

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv16x16(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=16, stride=stride, padding=0, bias=False)

#Region Score Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.public_conv = conv1x1(256, 128)
        self.bn = nn.BatchNorm2d(128)
        self.relu = F.relu
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.score_conv = conv1x1(128, 1)
        self.act_prob = torch.sigmoid
        self.MSE_loss = nn.MSELoss()

    def forward(self, x):
        feature = self.public_conv(x)
        feature = self.bn(feature)
        feature = self.relu(feature)
        score_feature = self.pooling(feature)
        score = self.score_conv(score_feature)
        score = self.act_prob(score)
        return [score,None]

    def loss_function(self, output, target):
        [output_score, _] = output
        [target_score,_] = target
        target_score = target_score * torch.ones(output_score.shape).float()
        score_loss = self.MSE_loss(output_score.to(cfg['device']), target_score.to(cfg['device']))
        loss = score_loss
        return loss