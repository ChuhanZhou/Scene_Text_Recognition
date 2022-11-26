import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, in_num=1, out_num=2):
        super().__init__()
        self.relu = F.relu
        if out_num == 2:
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.out_num = out_num
        if in_num == 2:
            self.conv2 = conv1x1(in_channels*2, out_channels)
        else:
            self.conv2 = conv1x1(in_channels, out_channels)
        self.conv3 = conv3x3(out_channels, out_channels)

    def forward(self, x1, x2=None):

        #x1 = self.conv1(x1)
        #x1 = self.bn(x1)
        #x1 = self.relu(x1)

        # 融合特征
        if x2 is not None:

            f = torch.cat([x1, x2], dim=1)
        else:
            f = x1

        p = self.conv2(f)
        p = self.bn(p)
        p = self.relu(p)

        p = self.conv3(p)
        p = self.bn(p)
        p = self.relu(p)

        if self.out_num == 2:
            f = self.up(p)
            return p, f
        return p


class Net(nn.Module):
    def __init__(self, in_channel_list):
        super().__init__()
        self.up1 = Upsample(in_channel_list[0], in_channel_list[1], 1, 2)
        self.up2 = Upsample(in_channel_list[1], in_channel_list[2], 2, 2)
        self.up3 = Upsample(in_channel_list[2], in_channel_list[3], 2, 2)
        #self.up4 = Upsample(in_channel_list[3], 256, 2, 2)

    def forward(self, x, down_f,out_p = True):
        p1, f1 = self.up1(x)
        p2, f2 = self.up2(down_f[-1],f1)
        p3, f3 = self.up3(down_f[-2],f2)

        #p4, f4 = self.up4(down_f[-3],f3)
        #if out_p:
        #    return f4,[p1,p2,p3,p4]
        #return f4

        if out_p:
            return f3, [p1, p2, p3]
        return f3

