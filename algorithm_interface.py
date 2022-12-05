import math

import cv2
import numpy as np
import datetime

import torch.nn as nn
import torch.utils.data

from configs.common_config import config as cfg
from tool import process_data
from tool import mark_image
from models import main_net

max_length = 2048
model = main_net.Net()
model_ckpt = torch.load(cfg["test_model"],map_location=torch.device(cfg['device']))
model.load_state_dict(model_ckpt, strict=False)
model.to(cfg['device'])

def recognize_text(image):
    if max(image.shape[1], image.shape[0]) > max_length:
        max_l = max(image.shape[1], image.shape[0])
        resize = [int(image.shape[0] * max_length / max_l) / image.shape[0],
                  int(image.shape[1] * max_length / max_l) / image.shape[1]]
        image = cv2.resize(image, (int(image.shape[1] * resize[1]), int(image.shape[0] * resize[0])),
                           interpolation=cv2.INTER_CUBIC)

    add_w = math.ceil(image.shape[1] / 32) * 32 - image.shape[1]
    add_h = math.ceil(image.shape[0] / 32) * 32 - image.shape[0]

    img = cv2.copyMakeBorder(image, 0, add_h, 0, add_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    data = process_data.img_to_data(img).unsqueeze(0)
    out = model(data)

    [ldn_decode, rsn_decode, ocr_decode] = process_data.decode_data_label(out)
    points_list, [dim_0_np, dim_1_np, result] = ldn_decode
    [points_score_list] = rsn_decode
    [text_list] = ocr_decode

    image_r = process_data.data_to_img(data)
    text_out_list = []
    for i,points_score in enumerate(points_score_list):
        [xywhr, box_points, _, score] = points_score
        image_r, _ = mark_image.mark_image_by_box(image_r, box_points,x_y_trans=False,horizontal=False)
        text = "[No Text]"
        ocr_xywhr_list = [xywhr for [xywhr, _] in text_list]
        if xywhr in ocr_xywhr_list:
            x, y, w, h, _ = xywhr
            text = text_list[ocr_xywhr_list.index(xywhr)][1]
            is_new = True
            for part in text_out_list:
                last_line = part[-1]
                l_x, l_y, l_w, l_h, _ = last_line[0]
                l_box_points = last_line[1]

                box_points_calculate = np.ones((4*4,2))
                l_box_points_calculate = np.ones((4*4,2))
                for n in range(4):
                    box_points_calculate[n*4:(n+1)*4,:] = box_points_calculate[n*4:(n+1)*4,:]*box_points[n:n+1,:]
                    l_box_points_calculate[n*4:(n+1)*4,:] = l_box_points_calculate[n*4:(n+1)*4,:]*l_box_points
                min_distance = np.min(np.sqrt(np.sum((box_points_calculate-l_box_points_calculate)**2,axis=1)))
                min_distance = min(min_distance,process_data.get_length([x,y],[l_x,l_y]))

                add = min(w, h) * cfg['wh_add']
                l_add = min(l_h, l_w) * cfg['wh_add']
                if min_distance <= (min(l_h, l_w) + l_add + min(h, w) + add) * 2 /2:
                    part.append([xywhr,box_points, text])
                    is_new = False
                    break

            if is_new:
                text_out_list.append([[xywhr,box_points, text]])

        #print(score,text)

    text_out = ""
    for i,part in enumerate(text_out_list):
        line_text = ""
        if i!=0:
            line_text += "\n\n"
        for p_i,[_,_,line] in enumerate(part):
            line_text += line
            if p_i != len(part)-1:
                line_text += "\n"
        text_out += line_text

    cv2.imwrite("out/r.png", image_r)
    cv2.imwrite("static/images/r.png", image_r)

    return text_out

if __name__ == '__main__':
    start_time = datetime.datetime.now()

    image = cv2.imread("test/Jose Harper.png")
    print(recognize_text(image))

    end_time = datetime.datetime.now()
    print((end_time - start_time))
