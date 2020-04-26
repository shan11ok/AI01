#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import cv2 as cv
from timeit import default_timer as timer

import numpy as np
from PIL import Image
from myutil.Util import Util

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
print(tf.__version__)
#查询tensorflow安装路径为:
print( tf.__path__ )
import keras.backend.tensorflow_backend as KTF
from keras.applications.inception_v3 import preprocess_input
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True   #不全部占满显存, 按需分配
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.Session(config=config)
KTF.set_session(sess)  # 设置session

#car_list = ['丰田C-HR', '五菱宏光', '大众探岳', '日产奇骏', '比亚迪E5', '荣威RX5', '马自达CX-5']
car_dict = Util.recover_dict(f"dict.txt")

from identification_model import New_Session_Model
model = New_Session_Model("models/vehicle_iden_model_iv3_ft.h5")


def resize_img_keep_ratio(img, target_size):
    '''
    保持比例缩放图片
    :param img: 图形数组
    :param target_size:
    :return:
    '''
    import cv2
    old_size = img.shape[0:2]
    #ratio = min(float(target_size)/(old_size))
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i*ratio) for i in old_size])
    img = cv2.resize(img,(new_size[1], new_size[0]))
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h//2, pad_h-(pad_h//2)
    left, right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0))
    return img_new


def car_identity(image):
    '''
    车型识别
    :param img:
    :return: 车型型号, 概率
    '''
    image_resized = image.resize((299, 299))
    cropImg_BGR = cv.cvtColor(np.asarray(image_resized), cv.COLOR_RGB2BGR)
    cropImg_pre_process = preprocess_input(np.expand_dims(cropImg_BGR, axis=0))
    result = model.predict(cropImg_pre_process)
    decoded = np.argmax(result, axis=1)[0]
    possiblity = np.max(result, axis=1)
    car_name = car_dict[decoded]
    print(f"{car_name} {possiblity}")
    return car_name, possiblity


def detect_video(yolo, video_path, output_path=""):
    '''
    检测视频中的车辆型号
    :param yolo: yolo模型的参数
    :param video_path: 输入视频路径
    :param output_path: 输出视频路径（可选）
    :return:
    '''
    import cv2
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        return_value, frame = vid.read()
        if frame is None:
            break
        image = Image.fromarray(frame)
        image = yolo.detect_image(image, car_identity)
        result = np.asarray(image)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


if __name__ == '__main__':
    from yolo import YOLO
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--input", nargs='?', type=str, required=True, help="Video input path")
    parser.add_argument("--output", nargs='?', type=str, required=False, default="", help="Video onput path")
    FLAGS = parser.parse_args()
    input_video = FLAGS.input
    print(f"Input video: {input_video}")
    if not os.path.exists(input_video):
        print(f"{input_video} is not exists!")
    else:
        detect_video(YOLO(**vars(FLAGS)), input_video, FLAGS.output)

