import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv7x7(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)


class Basic(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv1 = conv7x7(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = F.relu
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        return out


class Bottleneck(nn.Module):
    # stride=None:Identity_Block stride!=None:Conv_Block
    n = 4

    def __init__(self, in_channels, middle_channels=None, out_channels=None, stride=None):
        super().__init__()
        if stride is None:
            if middle_channels is None:
                middle_channels = int(in_channels / self.n)
            if out_channels is None:
                out_channels = in_channels
            stride = 1
            self.conv4 = None
            self.info = "[Identity_Block] in:{} middle:{} out:{}".format(in_channels, middle_channels, out_channels)
        else:
            if middle_channels is None:
                middle_channels = int(in_channels / 2)
            if out_channels is None:
                out_channels = middle_channels * self.n
            self.conv4 = conv1x1(in_channels, out_channels, stride)
            self.bn4 = nn.BatchNorm2d(out_channels)
            self.info = "[Conv_Block] in:{} middle:{} out:{}".format(in_channels, middle_channels, out_channels)

        self.relu = F.relu
        self.conv1 = conv1x1(in_channels, middle_channels)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = conv3x3(middle_channels, middle_channels, stride)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.conv3 = conv1x1(middle_channels, out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        #print(self.info)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.conv4 is not None:
            identity = self.conv4(identity)
            identity = self.bn4(identity)

        out += identity

        out = self.relu(out)

        return out


class Net(nn.Module):
    resnet_type = {"50": [3, 4, 6, 3], "101": [3, 4, 23, 3]}

    def __init__(self, type="50", in_dim = 3, dim_type=None):
        super().__init__()
        #print("resnet-{} init".format(type))
        if dim_type is None:
            dim_type = [64, 256, 512, 1024, 2048]
        self.setting = self.resnet_type.get(type)
        self.layer0 = Basic(in_dim)
        self.layer1_0 = Bottleneck(dim_type[0], middle_channels=dim_type[0], stride=1)
        self.layer1_1 = Bottleneck(dim_type[1])
        self.layer2_0 = Bottleneck(dim_type[1], stride=2)
        self.layer2_1 = Bottleneck(dim_type[2])
        self.layer3_0 = Bottleneck(dim_type[2], stride=2)
        self.layer3_1 = Bottleneck(dim_type[3])
        self.layer4_0 = Bottleneck(dim_type[3], stride=2)
        self.layer4_1 = Bottleneck(dim_type[4])

    def encode(self, x, out_layer = -1):
        out_list = []
        out = self.layer0(x)

        if out_layer == 0:
            return out, out_list

        out = self.layer1_0(out)
        for i in range(self.setting[0] - 1):
            out = self.layer1_1(out)
        out_list.append(out)

        if out_layer == 1:
            return out, out_list

        out = self.layer2_0(out)
        for i in range(self.setting[1] - 1):
            out = self.layer2_1(out)
        out_list.append(out)

        if out_layer == 2:
            return out, out_list

        out = self.layer3_0(out)
        for i in range(self.setting[2] - 1):
            out = self.layer3_1(out)
        out_list.append(out)

        if out_layer == 3:
            return out, out_list

        out = self.layer4_0(out)
        for i in range(self.setting[3] - 1):
            out = self.layer4_1(out)
        return out,out_list

    def forward(self, x,out_layer = -1):
        out,encode_out_list = self.encode(x,out_layer)
        return out,encode_out_list
