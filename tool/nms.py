import os
import cv2
import numpy as np
import math

import torch
import torch.nn as nn
import torchvision
from sklearn.preprocessing import MinMaxScaler

from configs.common_config import config as cfg
import shapely
from shapely.geometry import Polygon, MultiPoint
#from tool.kf_iou_loss import kfiou_loss


# iou type: kf_iou
def IoU_torch(pred, target, pred_decode=None, targets_decode=None,eps=1e-6):
    # (x,y,w,h,r) shape: (x*y,5)
    if pred_decode == None:
        pred_decode = pred
    if targets_decode == None:
        targets_decode = target
    return kfiou_loss(pred, target, pred_decode, targets_decode,eps=eps)


def DIoU_torch(output, target,eps=1e-6):
    # (x,y,w,h,r) shape: (5,y,x)
    x_length = target.shape[2]
    y_length = target.shape[1]
    theta_trans = torch.pi / 180

    output_box = torch.stack([target[0],
                              target[1],
                              target[2],
                              target[3],
                              target[4]], 2).reshape(-1, 5)
    target_box = torch.stack([output[0],
                              output[1],
                              output[2],
                              output[3],
                              output[4]], 2).reshape(-1, 5)
    output_box_dc = torch.stack([target[0] * x_length,
                                 target[1] * y_length,
                                 target[2] * x_length,
                                 target[3] * y_length,
                                 target[4] * 90], 2).reshape(-1, 5)
    target_box_dc = torch.stack([output[0] * x_length,
                                 output[1] * y_length,
                                 output[2] * x_length,
                                 output[3] * y_length,
                                 output[4] * 90], 2).reshape(-1, 5)

    iou_loss = IoU_torch(output_box, target_box, output_box_dc, target_box_dc,eps)

    t_center = [target_box_dc[:, 0],target_box_dc[:, 1]]
    o_center = [output_box_dc[:, 0],output_box_dc[:, 1]]

    t_w = target_box_dc[:, 2]
    t_h = target_box_dc[:, 3]
    t_theta = target_box_dc[:, 4]

    t_length = torch.sqrt((t_h / 2) ** 2 + (t_w / 2) ** 2)
    t_tan_o = (t_h / 2) / (t_w.clamp(min=eps) / 2)
    t_inv = torch.arctan(t_tan_o)
    t_theta_o = torch.rad2deg(t_inv)

    t_lt_r = (180 + t_theta_o + t_theta) * theta_trans
    t_rt_r = (360 - t_theta_o + t_theta) * theta_trans
    t_lb_r = (t_theta_o + t_theta) * theta_trans
    t_rb_r = (180 - t_theta_o + t_theta) * theta_trans

    o_w = output_box_dc[:, 2]
    o_h = output_box_dc[:, 3]
    o_theta = output_box_dc[:, 4]

    o_length = torch.sqrt((o_h / 2) ** 2 + (o_w / 2) ** 2)
    o_tan_o = (o_h / 2) / (o_w.clamp(min=eps) / 2)
    o_inv = torch.arctan(o_tan_o)
    o_theta_o = torch.rad2deg(o_inv)

    o_lt_r = (180 + o_theta_o + o_theta) * theta_trans
    o_rt_r = (360 - o_theta_o + o_theta) * theta_trans
    o_lb_r = (o_theta_o + o_theta) * theta_trans
    o_rb_r = (180 - o_theta_o + o_theta) * theta_trans

    p_x = torch.cat([torch.stack([torch.sin(t_lt_r) * t_length + t_center[0]]),
                     torch.stack([torch.sin(t_rt_r) * t_length + t_center[0]]),
                     torch.stack([torch.sin(t_lb_r) * t_length + t_center[0]]),
                     torch.stack([torch.sin(t_rb_r) * t_length + t_center[0]]),
                     torch.stack([torch.sin(o_lt_r) * o_length + o_center[0]]),
                     torch.stack([torch.sin(o_rt_r) * o_length + o_center[0]]),
                     torch.stack([torch.sin(o_lb_r) * o_length + o_center[0]]),
                     torch.stack([torch.sin(o_rb_r) * o_length + o_center[0]])],dim = 0)

    p_y = torch.cat([torch.stack([torch.cos(t_lt_r) * t_length + t_center[1]]),
                     torch.stack([torch.cos(t_rt_r) * t_length + t_center[1]]),
                     torch.stack([torch.cos(t_lb_r) * t_length + t_center[1]]),
                     torch.stack([torch.cos(t_rb_r) * t_length + t_center[1]]),
                     torch.stack([torch.cos(o_lt_r) * o_length + o_center[1]]),
                     torch.stack([torch.cos(o_rt_r) * o_length + o_center[1]]),
                     torch.stack([torch.cos(o_lb_r) * o_length + o_center[1]]),
                     torch.stack([torch.cos(o_rb_r) * o_length + o_center[1]])],dim = 0)

    p_min = [torch.min(p_x,dim = 0)[0],torch.min(p_y,dim = 0)[0]]
    p_max = [torch.max(p_x, dim=0)[0], torch.max(p_y, dim=0)[0]]

    rho = torch.sqrt((t_center[0] - o_center[0]) ** 2 + (t_center[1] - o_center[1]) ** 2)
    c = torch.sqrt((p_max[0] - p_min[0]) ** 2 + (p_max[1] - p_min[1]) ** 2) #有待验证是否是最小闭包区域对角线长度
    diou = iou_loss + (rho ** 2) / (c.clamp(min=eps) ** 2)
    diou = torch.stack([diou])
    return diou


def DIoU_torch_single(p, target, output):
    diou = 0
    if p == 1:
        r_transform = torch.pi / 180
        [t_center, t_w, t_h, t_theta] = target
        [o_center, o_w, o_h, o_theta] = output

        t_length = torch.sqrt((t_h / 2) ** 2 + (t_w / 2) ** 2)
        t_tan_o = (t_h / 2) / (t_w / 2)
        t_inv = torch.arctan(t_tan_o)
        t_theta_o = torch.rad2deg(t_inv)

        t_lt_r = (180 + t_theta_o + t_theta) * r_transform
        t_rt_r = (360 - t_theta_o + t_theta) * r_transform
        t_lb_r = (t_theta_o + t_theta) * r_transform
        t_rb_r = (180 - t_theta_o + t_theta) * r_transform

        t_lt = torch.stack([(torch.sin(t_lt_r) * t_length + t_center[0]), (torch.cos(t_lt_r) * t_length + t_center[1])],0)
        t_rt = torch.stack([(torch.sin(t_rt_r) * t_length + t_center[0]), (torch.cos(t_rt_r) * t_length + t_center[1])],0)
        t_lb = torch.stack([(torch.sin(t_lb_r) * t_length + t_center[0]), (torch.cos(t_lb_r) * t_length + t_center[1])],0)
        t_rb = torch.stack([(torch.sin(t_rb_r) * t_length + t_center[0]), (torch.cos(t_rb_r) * t_length + t_center[1])],0)
        t_box = torch.stack([t_lt, t_rt, t_lb, t_rb], 0)

        o_length = torch.sqrt((o_h / 2) ** 2 + (o_w / 2) ** 2)
        o_tan_o = (o_h / 2) / (o_w / 2)
        o_inv = torch.arctan(o_tan_o)
        o_theta_o = torch.rad2deg(o_inv)
        o_lt_r = (180 + o_theta_o + o_theta) * r_transform
        o_rt_r = (360 - o_theta_o + o_theta) * r_transform
        o_lb_r = (o_theta_o + o_theta) * r_transform
        o_rb_r = (180 - o_theta_o + o_theta) * r_transform
        o_lt = torch.stack([(torch.sin(o_lt_r) * o_length + o_center[0]), (torch.cos(o_lt_r) * o_length + o_center[1])],0)
        o_rt = torch.stack([(torch.sin(o_rt_r) * o_length + o_center[0]), (torch.cos(o_rt_r) * o_length + o_center[1])],0)
        o_lb = torch.stack([(torch.sin(o_lb_r) * o_length + o_center[0]), (torch.cos(o_lb_r) * o_length + o_center[1])],0)
        o_rb = torch.stack([(torch.sin(o_rb_r) * o_length + o_center[0]), (torch.cos(o_rb_r) * o_length + o_center[1])],0)
        o_box = torch.stack([o_lt, o_rt, o_lb, o_rb], 0)

        p_min = [min(torch.min(t_box[:, 0]), torch.min(o_box[:, 0])),
                 min(torch.min(t_box[:, 1]), torch.min(o_box[:, 1]))]
        p_max = [max(torch.max(t_box[:, 0]), torch.max(o_box[:, 0])),
                 max(torch.max(t_box[:, 1]), torch.max(o_box[:, 1]))]

        iou = 0  # IoU_torch(a, b)
        rho = torch.sqrt((t_center[0] - o_center[0]) ** 2 + (t_center[1] - o_center[1]) ** 2)
        c = torch.sqrt((p_max[0] - p_min[0]) ** 2 + (p_max[1] - p_min[1]) ** 2)

        diou = 1 - iou + (rho ** 2) / (c ** 2)
        if diou > 2:
            diou = diou / diou * 3
    return diou


def IoU(points_a, points_b):
    poly_a = Polygon(np.array(points_a)).convex_hull

    poly_b = Polygon(np.array(points_b)).convex_hull

    iou = 0
    ioa = 0
    iob = 0
    if poly_a.intersects(poly_b):
        inter_area = poly_a.intersection(poly_b).area
        union_area = poly_a.area + poly_b.area - inter_area
        if union_area != 0:
            iou = inter_area / union_area
            ioa = inter_area/poly_a.area
            iob = inter_area/poly_b.area
    return [iou,ioa,iob]


def DIoU(p, a1_p1_x, a1_p1_y, a1_p2_y, a1_p2_x, a1_p3_y, a1_p3_x, a1_p4_y, a1_p4_x, a2_p1_y, a2_p1_x, a2_p2_y,
         a2_p2_x, a2_p3_y, a2_p3_x, a2_p4_y, a2_p4_x):
    diou = 0
    if p == 1:
        a = [[a1_p1_y, a1_p1_x],
             [a1_p2_y, a1_p2_x],
             [a1_p3_y, a1_p3_x],
             [a1_p4_y, a1_p4_x]]
        b = [[a2_p1_y, a2_p1_x],
             [a2_p2_y, a2_p2_x],
             [a2_p3_y, a2_p3_x],
             [a2_p4_y, a2_p4_x]]

        center_a = Polygon(np.array(a)).centroid
        center_b = Polygon(np.array(b)).centroid

        p_min = [min(np.min(np.array(a)[:, 0]), np.min(np.array(b)[:, 0])),
                 min(np.min(np.array(a)[:, 1]), np.min(np.array(b)[:, 1]))]
        p_max = [max(np.max(np.array(a)[:, 0]), np.max(np.array(b)[:, 0])),
                 max(np.max(np.array(a)[:, 1]), np.max(np.array(b)[:, 1]))]

        [iou,ioa,iob] = IoU(a, b)
        rho = math.sqrt((center_a.x - center_b.x) ** 2 + (center_a.y - center_b.y) ** 2)
        c = math.sqrt((p_max[0] - p_min[0]) ** 2 + (p_max[1] - p_min[1]) ** 2)

        diou = 1 - iou + (rho ** 2) / (c ** 2)
        if diou > 2:
            diou = 3
    return diou
