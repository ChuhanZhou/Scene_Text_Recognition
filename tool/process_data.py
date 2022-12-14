import os
import cv2
import numpy as np
import math
from xml.dom import minidom

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torchvision.transforms import functional as tf

from configs.common_config import config as cfg
from tool import mark_image
from tool import nms


def read_identification_dict():
    path = cfg["identification_list"]
    identification_dict = {"blank":0, " ":1,"#":2}
    if has_file(path):
        file = open(path, "r").readlines()
        for signs in file:
            sign_list = signs.split("\n")[0]
            for sign in sign_list:
                identification_dict[sign] = len(identification_dict)
    return identification_dict


def read_data(run_type, create_data_label=False):
    dataset_path_list = cfg["{}_dataset".format(run_type)]
    dataset_mix = []
    for dataset_i, dataset_path in enumerate(dataset_path_list):
        dataset = []
        label_point_num = cfg["{}_label_point_num".format(run_type)][dataset_i]
        label_point_style = cfg["{}_label_point_style".format(run_type)][dataset_i]
        label_point_order = cfg["{}_label_point_order".format(run_type)][dataset_i]
        list_file_name = "image_label_list.txt"
        list_file_path = "{}/{}".format(dataset_path, list_file_name)
        if not has_file(list_file_path):
            create_link_file(run_type, dataset_path, list_file_name, dataset_i)
        image_label_list = open(list_file_path, "r").readlines()
        max_length = cfg['input_max_length']
        max_image_num = cfg["{}_max_image_num".format(run_type)][dataset_i]

        if max_image_num > 0:
            print("loading dataset [{}]".format(dataset_path))
            for i, image_label in enumerate(image_label_list):
                image_label = image_label.split('\n')[0]

                name_parts = image_label.split("/")[-1].split(".")
                name = ""
                for i, p in enumerate(name_parts):
                    if i == 0:
                        name += p
                    elif i != len(name_parts) - 1:
                        name += ".{}".format(p)

                if cfg['input_dim'] == 1:
                    image = cv2.imread(image_label.split(" ")[0], cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(image_label.split(" ")[0])

                if image is not None:
                    resize = [1, 1]  # [y,x]
                    if max(image.shape[1], image.shape[0]) > max_length:
                        max_l = max(image.shape[1], image.shape[0])
                        resize = [int(image.shape[0] * max_length / max_l) / image.shape[0],
                                  int(image.shape[1] * max_length / max_l) / image.shape[1]]
                        image = cv2.resize(image, (int(image.shape[1] * resize[1]), int(image.shape[0] * resize[0])),
                                           interpolation=cv2.INTER_CUBIC)

                    # 检查图片大小，保证边长可以被32(2^5)整除
                    add_w = math.ceil(image.shape[1] / 32) * 32 - image.shape[1]
                    add_h = math.ceil(image.shape[0] / 32) * 32 - image.shape[0]
                    # add_w = max_length - image.shape[1]
                    # add_h = max_length - image.shape[0]
                    image = cv2.copyMakeBorder(image, 0, add_h, 0, add_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

                    data = img_to_data(image)

                    label = read_label(open(image_label.split(" ")[1], "r", encoding='UTF-8').readlines(),label_point_num,label_point_style,label_point_order)

                    data_label = "None"
                    if create_data_label:
                        data_label, label = encode_data_label(image, label, resize, need_unsqueeze=False)
                    dataset.append([name, data, label, data_label, resize])

                    n = len(dataset)
                    print_p = math.ceil(max_image_num / 4)
                    if int(n / print_p) * print_p == n or n == max_image_num:
                        print("{}/{}".format(n, max_image_num))
                if len(dataset) >= max_image_num:
                    break

        dataset_mix += dataset
    return dataset_mix

def read_label(label_text, label_point_num,label_point_style,label_point_order):
    # out:
    # points:[up_start,...,up_end,bottom_end,...,bottom_start]
    # point:[y,x]
    label_list = []
    for n, text in enumerate(label_text):
        if len(text.split(",")) > 1:
            text_sp = text.split("\n")[0].split(",")
            label = "###"
            if len(text_sp) >= (label_point_num * 2 + 1):
                label = text_sp[label_point_num * 2]
            for i, l_part in enumerate(text_sp):
                if i > label_point_num * 2:
                    label += ",{}".format(l_part)

            coordinates = text.split(",{}\n".format(label))[0].split(",")
            points_read = []
            for i, coordinate in enumerate(coordinates):
                index = int(i / 2)

                coordinate = int(float(coordinate))
                if index * 2 == i: #i:0,2,4...
                    points_read.append([coordinate])
                else: #i:1,3,5...
                    if label_point_style[1] == 0: #coordinate:x
                        points_read[index] = [points_read[index][0],coordinate]
                    else: #coordinate:y
                        points_read[index] = [coordinate, points_read[index][0]]

            order_key_list = list(label_point_order.keys())
            top_order = abs(order_key_list.index(0) - order_key_list.index(1))
            bottom_order = abs(order_key_list.index(2) - order_key_list.index(3))
            points = []
            o_l = [top_order,bottom_order]
            for i in range(2): #sort points
                p_i = i*2
                if o_l[i] == 1:
                    if order_key_list.index(p_i)<order_key_list.index(p_i+1):
                        points += points_read[label_point_order[p_i]:label_point_order[p_i+1]+1]
                    else:
                        points += list(reversed(points_read[label_point_order[p_i+1]:label_point_order[p_i]+1]))
                else:
                    if order_key_list.index(p_i)<order_key_list.index(p_i+1):
                        points += list(reversed(points_read[label_point_order[p_i+1]:len(label_point_order)+1] + points_read[0:label_point_order[p_i]+1]))
                    else:
                        points += points_read[label_point_order[p_i]:len(label_point_order)+1] + points_read[0:label_point_order[p_i+1]+1]

            if len(label)>0 and label[0] == label[-1] and label[0] == "\"":
                label = label[1:len(label) - 1]
            label_list.append([points, label])
    return label_list


def create_link_file(run_type, path, file_name, dataset_i):
    image_package = cfg["{}_image".format(run_type)][dataset_i]
    label_package = cfg["{}_label".format(run_type)][dataset_i]
    image_list = os.listdir("{}/{}".format(path, image_package))
    label_list = os.listdir("{}/{}".format(path, label_package))

    label_dict = {}
    for label_name in label_list:
        name = label_name.split(".txt")[0]
        label_dict[name] = label_name

    link_file = open("{}/{}".format(path, file_name), "w")
    for image_name in image_list:
        name_parts = image_name.split(".")
        name = ""
        for i, p in enumerate(name_parts):
            if i == 0:
                name += p
            elif i != len(name_parts) - 1:
                name += ".{}".format(p)
        if label_dict.__contains__(name):
            link = "{}/{}/{} {}/{}/{}\n".format(path, image_package, image_name, path, label_package, label_dict[name])
            link_file.write(link)
    link_file.close()


def has_file(file_path):
    if file_path == None:
        return False
    return os.path.isfile(file_path)


def img_to_data(img):
    dim = 1
    if len(img.shape) == 3:
        dim = img.shape[2]
    data = np.zeros((dim, img.shape[0], img.shape[1]))
    if len(img.shape) == 3:
        for i in range(dim):
            data[i] = img[:, :, i]
    else:
        data[0] = img[:, :]
    data = torch.tensor(data).float()
    return data


def data_to_img(data):
    img = None
    if data!=None:
        for i in range(4 - len(data.shape)):
            data = data.unsqueeze(0)
        data = data.detach().cpu().numpy()
        img = np.zeros((data.shape[2], data.shape[3], data.shape[1]))
        for i in range(data.shape[1]):
            img[:, :, i] = data[0][i]
    return img

def point_x_y_transform(point):
    return [point[1], point[0]]

def encode_data_label(img, label_list, resize=None, need_unsqueeze=True):
    if resize is None:
        resize = [1, 1]
    scaling = cfg['label_scaling']
    identification_dict = read_identification_dict()

    def point_value_resize(point):
        return [point[0] * resize[0], point[1] * resize[1]]

    def point_value_check(point,shape):
        return [min(max(point[0],0),shape[0]-1), min(max(point[1],0),shape[1]-1)]

    def create_key_points(start,end,total_num=7):
        p_0 = (end[0]-start[0])/(total_num-1)
        p_1 = (end[1]-start[1])/(total_num-1)
        points = [start]
        for i in range(total_num-2):
            points.append([points[i][0]+p_0,points[i][1]+p_1])
        points.append(end)
        points = np.round(points).astype(np.int).tolist()
        return points

    img_calculate = cv2.resize(img, (int(img.shape[1] / scaling), int(img.shape[0] / scaling)),
                               interpolation=cv2.INTER_CUBIC)

    data_label = np.zeros((img_calculate.shape[0], img_calculate.shape[1], 2))
    letter_label_map = np.zeros((img.shape[0], img.shape[1], 1))

    box_points_list = []
    xywht_text_list = []
    for i, label in enumerate(label_list):
        points = [point_value_resize(p) for p in label[0]]

        text = label[1]

        points = ((np.array(points) / scaling).astype(int)).tolist()
        points = [point_value_check(p, (img_calculate.shape[0], img_calculate.shape[1])) for p in points]

        t_points = points[0:int(len(points) / 2)]
        b_points = list(reversed(points[int(len(points) / 2):len(points)]))

        if len(points)==4:
            t_points = create_key_points(t_points[0],t_points[1])
            b_points = create_key_points(b_points[0],b_points[1])
            points = t_points + list(reversed(b_points))
        label_list[i][0] = points

        points_x_y = np.array([point_x_y_transform(p) for p in points])
        [[center_x, center_y], [w, h], theta] = cv2.minAreaRect(points_x_y)

        if w > 0 and h > 0:
            #box_key_points = get_points_of_box([center_x, center_y], [w, h], theta)

            h_mean_length = np.mean(np.sqrt((np.array(t_points)[:, 0:1] - np.array(b_points)[:, 0:1]) ** 2 + (
                        np.array(t_points)[:, 1:2] - np.array(b_points)[:, 1:2]) ** 2))
            w_mean_length = np.sum(
                np.sqrt((np.array(t_points)[0:len(t_points) - 1, 0:1] - np.array(t_points)[1:len(t_points), 0:1]) ** 2 +
                        (np.array(t_points)[0:len(t_points) - 1, 1:2] - np.array(t_points)[1:len(t_points),
                                                                        1:2]) ** 2) +
                np.sqrt((np.array(b_points)[0:len(b_points) - 1, 0:1] - np.array(b_points)[1:len(b_points), 0:1]) ** 2 +
                        (np.array(b_points)[0:len(b_points) - 1, 1:2] - np.array(b_points)[1:len(b_points),
                                                                        1:2]) ** 2)) / 2

            shrink_length = round(max(min(h_mean_length, w_mean_length) * 0.25, 1))

            points_x_y = np.array(points_x_y, np.int32).reshape((-1, 1, 2))
            dim_0 = data_label[:, :, 0:1].astype(np.uint8)
            dim_0 = cv2.fillPoly(dim_0, [points_x_y], (1))

            dim_1 = data_label[:, :, 1:2].astype(np.uint8)
            dim_1 = cv2.polylines(dim_1, [points_x_y], True, (1), shrink_length)

            dim_0_shrink = np.where(dim_1 > 0, 0, dim_0)
            dim_1 = np.where(dim_0 + dim_1 >= 1, dim_1, 0.5)
            data_label[:, :, 0:1] = dim_0_shrink  # 目标置信度 [0:1]
            data_label[:, :, 1:2] = dim_1  # 边界置信度 [0:1]

            box_points_list.append(points)

            text_label = []
            if text == "###":
                text_label.append(identification_dict["#"])
            else:
                for t_i,sign in enumerate(text):
                    index = -1
                    if sign in identification_dict.keys():
                        index = identification_dict[sign]
                        text_label.append(index)

                    else:
                        index = identification_dict["#"]
                        text_label.append(index)

            xywht_text_list.append([torch.tensor([center_x, center_y, w, h, theta]).float(), 1, torch.tensor(text_label),torch.tensor(points)])
            # data_label[:,:,2:6] = dim_2 #[左上,右上,右下,左下]关键点置信度 [0:1]

    ldn_data_label = img_to_data(data_label)
    rpn_data_label = [xywht_text_list]
    ocr_data_label = [img_to_data(letter_label_map)]

    if need_unsqueeze:
        ldn_data_label = ldn_data_label.unsqueeze(0)
        rpn_data_label[1] = rpn_data_label[1].unsqueeze(0)

    return [ldn_data_label, rpn_data_label, ocr_data_label], label_list

def encode_text_label(text,identification_dict=None):
    if identification_dict is None:
        identification_dict = read_identification_dict()
    text_label = []
    if text == "###":
        text_label.append(identification_dict["#"])
    else:
        for t_i, sign in enumerate(text):
            index = -1
            if sign in identification_dict.keys():
                index = identification_dict[sign]
                text_label.append(index)

            else:
                index = identification_dict["#"]
                text_label.append(index)
    return text_label

def decode_text_label(text_label,identification_dict = None):
    if identification_dict is None:
        identification_dict = read_identification_dict()
    text = ""
    for encode_text in text_label:
        decode = list(identification_dict.keys())[encode_text]
        text+=decode
    return text

def decode_data_label(data_label, p=0.85):
    ldn_decode = decode_ldn_data_label(data_label[0])
    rsn_decode = decode_rsn_data_label(data_label[1])
    ocr_decode = decode_ocr_data_label(data_label[2],is_final=True)
    return [ldn_decode, rsn_decode, ocr_decode]


def decode_ocr_data_label(data_label,is_final=True):
    identification_dict = read_identification_dict()
    text_list = []
    for [xywhr, score, ocr_out, text_label] in data_label:
        text = ""
        part = []
        new = False
        [cnn_out] = ocr_out
        out = cnn_out
        for i in range(out.shape[0]):
            probability_list = out[i, :].detach().cpu().numpy()#[1:len(identification_list)]
            decode_id = np.argmax(probability_list)#+1
            sign = list(identification_dict.keys())[decode_id]
            if decode_id != 0:
                if decode_id != 2:
                    if len(part) == 0 or sign != part[-1] or not is_final:
                        part.append(sign)
                        text += sign
                else:
                    part.append("#")
                    text += "#"
            elif decode_id == 0:
                if len(part) > 0:
                    part = []
                if not is_final:
                    text += "_"
        text_list.append([xywhr, text])
    return [text_list]


def decode_rsn_data_label(data_label):
    scaling = cfg['detection_box_scaling']
    encode_scaling = cfg['label_scaling']
    threshold = cfg['rsn_score_threshold']
    points_list = []
    for [xywhr,score,_,points_dim,_,_] in data_label:
        if score > threshold:
            score = np.mean(score.detach().cpu().numpy())
            [box_points,points] = decode_rsn_points_dim(xywhr, points_dim, scaling * encode_scaling)
            points_list.append([xywhr, box_points ,points, score])
    return [points_list]

def decode_rsn_points_dim(xywhr,points_dim,scaling=1):
    # out:
    # point[x,y]

    [x, y, w, h, theta] = xywhr
    add = min(w, h) * cfg['wh_add']
    area_points_x_y = np.array(get_points_of_box([x, y], [w, h], theta, add,scaling), np.int32)
    min_0 = max(np.min(area_points_x_y[:, 0]), 0)
    min_1 = max(np.min(area_points_x_y[:, 1]), 0)

    points = None
    if points_dim is not None:
        shape = points_dim.shape[2:4]
        points_dim = torch.reshape(points_dim, (points_dim.shape[1], points_dim.shape[2] * points_dim.shape[3]))
        points = torch.argmax(points_dim, dim=1, keepdim=False)
        points_x = points % shape[1]*scaling + min_0
        points_y = torch.torch.floor(points / shape[1]).long()*scaling + min_1
        points = torch.cat([points_x.unsqueeze(1), points_y.unsqueeze(1)], dim=1).detach().cpu().numpy().tolist()
    return [area_points_x_y,points]


def decode_ldn_data_label(data_label):
    scaling = cfg['detection_box_scaling']
    encode_scaling = cfg['label_scaling']

    data_label = data_to_img(data_label)
    data_label = cv2.resize(data_label, (int(data_label.shape[1] / scaling), int(data_label.shape[0] / scaling)),
                            interpolation=cv2.INTER_CUBIC)
    data_label = img_to_data(data_label).cpu().numpy()

    dim_0_np = data_label[0]
    dim_1_np = data_label[1]
    result = np.where(dim_0_np > dim_1_np, 1, 0)
    n, id_map, info_list, center = cv2.connectedComponentsWithStats(result.astype(np.uint8), connectivity=8)
    points_list = []
    for i, info in enumerate(info_list):
        area = info[4]
        if info[2] >= 3 and info[3] >= 2 and i != 0 or info[2] >= 2 and info[3] >= 3 and i != 0:
            p_list_i = torch.nonzero(torch.tensor(id_map) == i).cpu().numpy()
            [[center_y, center_x], [h, w], theta] = cv2.minAreaRect(p_list_i)
            theta = -theta
            add = max(min(w, h), 1) * cfg['wh_add']
            points = get_points_of_box([center_x, center_y], [w, h], theta, add, scaling * encode_scaling)
            points_list.append(points)

    if len(points_list) > 0:
        points_list = np.round(np.array(points_list)).astype(int).tolist()
    return points_list, [dim_0_np, dim_1_np, result]


def get_points_of_box(center, box_shape, theta, padding=0, scaling=1):
    # shape [w,h] or [h,w]
    # center [x,y] or [y,x]
    # box_shape [w,h] or [h,w]
    k1 = 1.1
    k2 = 1.1
    #if min(box_shape[0],box_shape[1])<6 and min(box_shape[0],box_shape[1])>0:
    #    if box_shape[0]>box_shape[1]:
    #        k1 = min(max(box_shape[0],box_shape[1])/min(box_shape[0],box_shape[1])*(6-min(box_shape[0],box_shape[1]))/6,3)
    #        k2 = min(max(box_shape[0],box_shape[1])/min(box_shape[0],box_shape[1])*(6-min(box_shape[0],box_shape[1]))/6,1.5)
    #    else:
    #        k1 = min(max(box_shape[0], box_shape[1]) / min(box_shape[0], box_shape[1])*(6-min(box_shape[0],box_shape[1]))/6, 1.5)
    #        k2 = min(max(box_shape[0], box_shape[1]) / min(box_shape[0], box_shape[1])*(6-min(box_shape[0],box_shape[1]))/6, 3)
    w = max(box_shape[0] + padding*k1, 1)
    h = max(box_shape[1] + padding*k2, 1)
    center_x = center[0]
    center_y = center[1]
    length = math.sqrt((h / 2) ** 2 + (w / 2) ** 2)
    tan_o = (h / 2) / (w / 2)
    inv = np.arctan(tan_o)
    t_o = np.degrees(inv)
    r_transform = np.pi / 180
    lt_r = (180 + t_o + theta) * r_transform
    rt_r = (360 - t_o + theta) * r_transform
    lb_r = (180 - t_o + theta) * r_transform
    rb_r = (t_o + theta) * r_transform
    lt = [(np.cos(lt_r) * length + center_x) * scaling, (np.sin(lt_r) * length + center_y) * scaling]
    rt = [(np.cos(rt_r) * length + center_x) * scaling, (np.sin(rt_r) * length + center_y) * scaling]
    lb = [(np.cos(lb_r) * length + center_x) * scaling, (np.sin(lb_r) * length + center_y) * scaling]
    rb = [(np.cos(rb_r) * length + center_x) * scaling, (np.sin(rb_r) * length + center_y) * scaling]
    if theta > 45:
        return [lb, lt, rt, rb]
    elif theta < -45:
        return [rt, rb, lb, lt]
    else:
        return [lt, rt, rb, lb]

def get_length(point1, point2):
    return math.sqrt(np.square(abs(point1[0] - point2[0])) + np.square(abs(point1[1] - point2[1])))

def key_points_align (x,points):
    # input:
    # p[x,y]
    t_points_list = points[0:int(len(points) / 2)]
    b_points_list = list(reversed(points[int(len(points) / 2):len(points)]))
    t_points = torch.Tensor(np.array(t_points_list))
    b_points = torch.Tensor(np.array(b_points_list))
    mean_h = torch.round(torch.mean(torch.sqrt(torch.pow(t_points[:,0]-b_points[:,0],2)+torch.pow(t_points[:,1]-b_points[:,1],2)))).long().cpu().numpy().tolist()
    w_t = torch.sqrt(torch.pow(t_points[0:len(t_points)-1,0]-t_points[1:len(t_points),0],2)+torch.pow(t_points[0:len(t_points)-1,1]-t_points[1:len(t_points),1],2))
    w_b = torch.sqrt(torch.pow(b_points[0:len(b_points)-1,0]-b_points[1:len(b_points),0],2)+torch.pow(b_points[0:len(b_points)-1,1]-b_points[1:len(b_points),1],2))
    mean_w = torch.round((w_t+w_b)/2).long().cpu().numpy().tolist()
    align_list = []
    for i in range(len(t_points_list)-1):
        t_1 = t_points_list[i]
        t_2 = t_points_list[i+1]
        b_1 = b_points_list[i]
        b_2 = b_points_list[i+1]

        before = [t_1,t_2,b_1,b_2]
        before_uni = [list(p) for p in set(tuple(_) for _ in before)]
        if len(before_uni)>=4:
            after = [[0,0],[mean_w[i],0],[0,mean_h],[mean_w[i],mean_h]]
            part = tf.perspective(x,before,after,interpolation=tf.InterpolationMode.BILINEAR)[:,:,0:mean_h+1,0:mean_w[i]+1]
            align_list.append(part)
    align_area = None
    if len(align_list)>0:
        align_area = torch.cat(align_list,dim=3)
    return align_area
