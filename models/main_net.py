import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

from configs.common_config import config as cfg
from models.backbone import resnet
from models.backbone import fpn
from models import text_line_detection_net as ldn
from models import rsn
from models import ocr_net
from tool import process_data
from tool import nms
from tool import mark_image

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet_type = cfg['resnet_type']
        feature_dim_list = [2048, 1024, 512, 256]
        self.identification_dict = process_data.read_identification_dict()
        cfg["ocr_out_dim"] = len(self.identification_dict)

        self.ldn = ldn.Net(resnet_type,feature_dim_list)
        self.rsn = rsn.Net()
        self.ocr_net = ocr_net.Net(3,cfg["ocr_out_dim"])

        self.ldn_loss = self.ldn.loss_function
        self.rsn_loss = self.rsn.loss_function
        self.ocr_loss = self.ocr_net.loss_function

    def forward(self, x):
        # 下采样 提取特征
        ldn_out,feature = self.ldn(x)

        area_list = self.get_area(ldn_out, max_area=-1)

        rsn_out = []
        for info in area_list:
            [xywhr,target_score,text_label,target_points] = info
            area_feature = self.get_feature_by_xywhr(xywhr, feature)
            [score,points_dim] = self.rsn(area_feature)
            rsn_out.append([xywhr,score,target_score,points_dim,None,text_label])

        threshold = cfg['rsn_score_threshold']
        ocr_out = []
        for i in range(len(rsn_out)):
            [xywhr, score, target_score,points_dim,target_points_dim,text_label] = rsn_out[i]
            if score >= threshold and target_score!=0 or target_score==1:
                if target_points_dim!=None:
                    points_dim = target_points_dim
                area_feature = self.get_feature_by_points_dim(xywhr, points_dim, x, 2)

                if area_feature!=None:
                    ocr_part_out = self.ocr_net(area_feature)
                    ocr_out.append([xywhr, score, ocr_part_out, text_label])
        return [ldn_out,rsn_out,ocr_out]

    def get_area(self,ldn_out,max_area=-1,score=-1.0,whitelist=None,blacklist=None):
        binary = torch.where(ldn_out[:,0:1]-ldn_out[:,1:2]>0,ldn_out[:,0:1]/ldn_out[:,0:1],ldn_out[:,0:1]*0)
        n, id_map, info_list, center = cv2.connectedComponentsWithStats(binary.detach().cpu().numpy()[0][0].astype(np.uint8), connectivity=8)

        id_dim = np.arange(0,n)[:,np.newaxis]
        info_list = np.concatenate((info_list,id_dim),axis=1)
        info_list = info_list[1:len(info_list)]

        if max_area != -1:
            info_list = info_list[np.argsort(info_list[:,4])[::-1]]

        area_list = []
        for i, info in enumerate(info_list):
            if info[2] >= 3 and info[3]>=2 or info[2] >= 2 and info[3]>=3:
                p_list_i = torch.nonzero(torch.tensor(id_map) == info[5]).numpy()
                [[y, x], [h, w], theta] = cv2.minAreaRect(p_list_i)
                theta = -theta

                if whitelist != None or blacklist != None:
                    max_iou = 0
                    max_ioa = 0
                    max_iob = 0
                    max_text = None
                    max_points = None
                    points = process_data.get_points_of_box([x,y],[w,h],theta)

                    if whitelist != None:
                        for [[b_x,b_y,b_w,b_h,b_theta],_,text,key_points] in whitelist:
                            w_points = process_data.get_points_of_box([b_x,b_y],[b_w,b_h],b_theta)
                            [iou,ioa,iob] = nms.IoU(points,w_points)
                            if iou > max_iou:
                                max_iou = iou
                                max_ioa = ioa
                                max_iob = iob
                                max_text = text
                                max_points = key_points
                        if max_ioa == 1 and max_iob >= 0.125:  # 检测框在目标框内：未完整识别一行（字间距过大）
                            area_list.append([[x, y, w, h, theta], score,max_text,max_points])
                        elif max_ioa >= 0.85 and max_iob >= 0.2:  # 检测框与目标框重叠：未完整识别一行（字间距过大），噪点干扰造成位移
                            area_list.append([[x, y, w, h, theta], score,max_text,max_points])

                    if blacklist != None:
                        for [[b_x,b_y,b_w,b_h,b_theta],_,_,_] in blacklist:
                            b_points = process_data.get_points_of_box([b_x,b_y],[b_w,b_h],b_theta)
                            [iou,ioa,iob] = nms.IoU(points,b_points)
                            if iou > max_iou:
                                max_iou = iou
                                max_ioa = ioa
                                max_iob = iob
                        if max_iou == 0: #检测框与目标框不重叠：未识别到目标
                            area_list.append([[x, y, w, h, theta], 0,None,None])
                        #else: #检测框与目标框小部分重叠：识别对象不一定是目标
                        #    area_list.append([[x, y, w, h, theta], 0])
                else:
                    area_list.append([[x,y,w,h,theta],score,None,None])

                if len(area_list)>=max_area and max_area!=-1:
                    break
        return area_list

    def get_feature_by_xywhr(self, xywhr, feature):
        [x,y, w,h, theta] = xywhr
        add = min(w, h) * cfg['wh_add'] #增大特征取值范围，获得更大视野
        points = process_data.get_points_of_box([x, y], [w, h], theta,add)
        points_x_y = np.array(points, np.int32)

        min_0 = max(np.min(points_x_y[:, 0]), 0)
        max_0 = min(np.max(points_x_y[:, 0]), feature.shape[3])
        min_1 = max(np.min(points_x_y[:, 1]), 0)
        max_1 = min(np.max(points_x_y[:, 1]), feature.shape[2])

        area_feature = feature[:, :, min_1:max_1 + 1, min_0:max_0 + 1]
        return area_feature

    def get_feature_by_points_dim(self,xywhr,points_dim,feature,scaling=1):
        area_points_x_y,points = process_data.decode_rsn_points_dim(xywhr,points_dim,scaling)
        area_feature = None

        if area_feature is None:
            area_feature = process_data.key_points_align(feature,area_points_x_y)
        return area_feature

    def loss_function(self, output, target,eps=1e-10):
        ldn_loss = self.ldn_loss(output[0], target[0])

        #rsn loss
        rsn_loss = 0
        rsn_output = output[1]
        for [_, output_score, target_score, output_points_dim, target_points_dim, _] in rsn_output:
            rsn_loss += self.rsn_loss([output_score, output_points_dim], [target_score,target_points_dim])
        if len(rsn_output) > 0:
            rsn_loss = rsn_loss / len(rsn_output)

        total_loss = 1*ldn_loss + 1*rsn_loss
        return total_loss,[1,len(rsn_output)],[ldn_loss,rsn_loss]
