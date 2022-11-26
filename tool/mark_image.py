import cv2
import numpy as np
from tool import process_data


def mark_image(image, point_list, p_c=[(0, 0, 255),(255,0 , 0),(0, 255,0),(0, 255, 255),(255, 255,0)], n=3, colors=(0, 0, 0),x_y_trans=True):
    # input: (x,y), x_y_trans=False
    # input: (y,x), x_y_trans=True
    def point_x_y_transform(point):
        return [point[1], point[0]]

    if x_y_trans:
        point_list = [point_x_y_transform(p) for p in point_list]

    result = image
    if len(image.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    for i, point in enumerate(point_list):
        if point[0] in range(image.shape[1]) and point[1] in range(image.shape[0]):
            y_s = max(point[1] - n + 1, 0)
            y_e = min(point[1] + n, result.shape[0] - 1)
            x_s = max(point[0] - n + 1, 0)
            x_e = min(point[0] + n, result.shape[1] - 1)

            result[y_s:y_e, x_s:x_e, 0] = p_c[i%len(p_c)][0]
            result[y_s:y_e, x_s:x_e, 1] = p_c[i%len(p_c)][1]
            result[y_s:y_e, x_s:x_e, 2] = p_c[i%len(p_c)][2]
            result[point[1]][point[0]][0] = colors[0]
            result[point[1]][point[0]][1] = colors[1]
            result[point[1]][point[0]][2] = colors[2]
    return result


def mark_image_by_box(image, point_list, l_c=(0, 255, 0), x_y_trans=True,horizontal=False):
    # input: (x,y), x_y_trans=False
    # input: (y,x), x_y_trans=True
    def point_x_y_transform(point):
        return [point[1], point[0]]

    result = np.array(image,np.uint8).copy()

    if len(image.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    if x_y_trans:
        point_list = [point_x_y_transform(p) for p in point_list]


    #p(x,y)
    #min_0 = max(np.min(np.array(point_list)[:, 0]),0)
    #max_0 = min(np.max(np.array(point_list)[:, 0]),image.shape[1])
    #min_1 = max(np.min(np.array(point_list)[:, 1]), 0)
    #max_1 = min(np.max(np.array(point_list)[:, 1]), image.shape[0])
    #lt = [min_0, min_1]
    #rt = [min_0, max_1]
    #lb = [max_0, min_1]
    #rb = [max_0, max_1]
    #if horizontal:
    #    point_list = [lt,rt,rb,lb]
    #p_x_y = np.array(point_list, np.int32).reshape((-1, 1, 2))
    #mask = cv2.fillPoly(np.zeros(result.shape), [p_x_y], (1,1,1))
    #part = (mask*result)[min_1:max_1+1,min_0:max_0+1,:]
    #result = cv2.polylines(result, [p_x_y], True, l_c,1)

    part = process_data.data_to_img(process_data.key_points_align(process_data.img_to_data(image).unsqueeze(0),point_list))
    p_x_y = np.array(point_list, np.int32).reshape((-1, 1, 2))
    result = cv2.polylines(result, [p_x_y], True, l_c, 1)

    return result,part
