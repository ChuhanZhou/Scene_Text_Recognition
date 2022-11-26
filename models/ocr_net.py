import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from models.backbone import resnet
from models.backbone import fpn
from configs.common_config import config as cfg
from tool import process_data

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv8x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=(8,1), stride=stride, padding=0, bias=False)

#Optical Character Recognition Network
#https://github.com/AstarLight/Lets_OCR/tree/master/recognizer/crnn
class Net(nn.Module):
    def __init__(self,in_channels=1, out_channels=1):
        super().__init__()
        self.resnet = resnet.Net(in_dim=in_channels)

        self.layer1_0 = conv3x3(512,256)
        self.layer1_1 = conv1x1(512,256)
        self.layer2_0 = conv3x3(512,256)
        self.layer2_1 = conv1x1(512,256)
        self.layer3_0 = conv3x3(512,256)
        self.layer3_1 = conv1x1(512,256)
        self.layer4_0 = conv3x3(512,256)
        self.layer4_1 = conv1x1(512,256)
        self.conv5 = conv8x1(512,out_channels)
        self.relu = F.relu
        self.bn = nn.BatchNorm2d(256)

        self.CTC_loss = nn.CTCLoss(blank=0, zero_infinity=True)


    def forward(self, x):
        x = F.interpolate(x, size=(64,math.ceil(x.shape[3]/x.shape[2]*64)))
        feature,_ = self.resnet(x,2)

        feature_half = self.layer1_0(feature)
        feature_half = self.bn(feature_half)
        feature = torch.cat([feature_half,self.bn(self.layer1_1(feature))], dim=1)
        feature = self.relu(feature)
        feature_half = self.layer2_0(feature)
        feature_half = self.bn(feature_half)
        feature = torch.cat([feature_half, self.bn(self.layer2_1(feature))], dim=1)
        feature = self.relu(feature)
        feature_half = self.layer3_0(feature)
        feature_half = self.bn(feature_half)
        feature = torch.cat([feature_half, self.bn(self.layer3_1(feature))], dim=1)
        feature = self.relu(feature)
        feature_half = self.layer4_0(feature)
        feature_half = self.bn(feature_half)
        feature = torch.cat([feature_half, self.bn(self.layer4_1(feature))], dim=1)
        feature = self.relu(feature)

        cnn_out = self.conv5(feature)
        cnn_out = cnn_out.squeeze(2)
        cnn_out = cnn_out.permute(2, 0, 1)

        return [cnn_out]

    def loss_function(self, output, target):
        loss = -1
        [cnn_output] = output
        input_length = torch.tensor([cnn_output.shape[0]], dtype=torch.int).to(cfg['device'])
        target_length = torch.tensor([target.shape[1]], dtype=torch.int).to(cfg['device'])
        if input_length[0] >= (target_length[0]*2+1):
            cnn_loss = self.CTC_loss(cnn_output.to(cfg['device']).log_softmax(2), target.to(cfg['device']), input_length, target_length)
            loss = cnn_loss
        return loss

    def get_letter_label(self, letter_map, shape,padding=0):
        stride = max(round(letter_map.shape[3] / shape[0]),1)
        scale = stride * shape[0] / letter_map.shape[3]
        letter_map = F.interpolate(letter_map, size=(round(scale*letter_map.shape[2]),round(scale*letter_map.shape[3])))
        max_pool = nn.MaxPool2d((letter_map.shape[2],stride),stride=stride)
        letter_label = torch.round(max_pool(letter_map)[0][0])
        for i in range(padding):
            left_add = torch.cat((letter_label[1:letter_label.shape[0]],torch.zeros((1)).to(cfg['device'])),0)
            letter_label = torch.where(letter_label!=0,letter_label,left_add)
            right_add = torch.cat((torch.zeros((1)).to(cfg['device']),letter_label[0:letter_label.shape[0]-1]),0)
            letter_label = torch.where(letter_label != 0, letter_label, right_add)
        start_i = -1
        for i in range(letter_label.shape[0]):
            if i>start_i:
                if letter_label[i]!=0 and letter_label[i]!=1:
                    start_i = i
                elif start_i is not None:
                    n = start_i+1
                    if letter_label[i]==1:
                        while n<letter_label.shape[0] and (letter_label[n]==0 or letter_label[n]== 1):
                            letter_label[n] = 1
                            start_i = n
                            n+=1
        return letter_label.long()

    def read_data(self,train=True,resize_num=[],dataset_num=-1):
        dataset_mix = []
        type = "train"
        if not train:
            type = "test"
        dataset_path_list = cfg["ocr_{}_dataset".format(type)]
        identification_dict = process_data.read_identification_dict()
        for i,dataset_path in enumerate(dataset_path_list):
            if i == dataset_num:
                break
            if len(resize_num)<=i:
                resize_num.append(1)
            else:
                resize_num[i] = max(resize_num[i],1)
            dataset = []
            print("loading dataset [{}]".format(dataset_path))
            list_file_name = cfg["ocr_{}_data_label".format(type)][i]
            list_file_path = "{}/{}".format(dataset_path, list_file_name)
            #if not process_data.has_file(list_file_path):
            #    self.create_link_file(dataset_path,list_file_path)
            image_label_list = open(list_file_path, "r").readlines()
            for image_label in image_label_list:
                info = image_label.split("\n")[0].split(" ")
                image_path = info[0]
                label = info[1]
                if len(info) > 2:
                    label_list = info[2:len(info)]
                    for part in label_list:
                        label += " {}".format(part)
                image = cv2.imread(image_path)
                h = 64
                resize = [h / image.shape[0],h / image.shape[0]]

                if resize[0] == 0 or resize[1] == 0:
                    print(image_path, resize)
                elif label != "#":
                    data_label = torch.tensor(np.array(process_data.encode_text_label(label, identification_dict)))
                    if np.mean(np.array(data_label))>2:
                        for n_i in range(resize_num[i]):
                            resize_i = [resize[0]/math.pow(2,n_i),resize[1]/math.pow(2,n_i)]
                            image_i = cv2.resize(image,(int(image.shape[1] * resize_i[1]), int(image.shape[0] * resize_i[0])),interpolation=cv2.INTER_CUBIC)
                            data_i = process_data.img_to_data(image_i)
                            dataset.append([image_path, data_i, data_label])
                            n = len(dataset)
                            print_p = math.ceil(len(image_label_list)*resize_num[i] / 4)
                            if int(n / print_p) * print_p == n or n == len(image_label_list)*resize_num[i]:
                                print("{}/{}".format(n, len(image_label_list)*resize_num[i]))
                #break
            dataset_mix+=dataset
        return dataset_mix