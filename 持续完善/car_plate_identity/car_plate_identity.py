'''
车牌检测程序Demo
参数：
--input： 检测图片路径
'''
from hyperlpr import pipline as pp
import cv2
import argparse
from timeit import default_timer as timer
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def identity_car_plate(img):
    '''
    检测车牌
    :param img: 图片数组
    :return:
    '''
    image, res = pp.SimpleRecognizePlate(img)
    #img_obj = Image.fromarray(img)
    #img_obj.show()
    return image, res


def process_image_file(image_file):
    '''
    检测车牌
    :param image_file: 图片路径
    :return:
    '''
    fontC = ImageFont.truetype("./font/platech.ttf", 32, 0);
    image = cv2.imread(image_file)
    image_array, res = identity_car_plate(image)
    img_obj = Image.fromarray(image_array)
    draw = ImageDraw.Draw(img_obj)
    x = 5
    y = 20
    for plate_str in res:
        plate_str = '车牌： ' + plate_str
        draw.text((x+2, y+2), plate_str, (0, 0, 0), font=fontC)
        draw.text((x, y), plate_str, (236, 28, 36), font=fontC)
        y = y + 38
    img_obj.show()
    return img_obj


def process_video(video_input, video_output=None):
    '''
    检测视频里的车牌
    :param video_input: 车辆视频路径
    :param video_output: 处理后的检测视频保存路径
    :return:
    '''
    is_write_video = False
    if video_output is not None:
        is_write_video = True

    import cv2
    vid = cv2.VideoCapture(video_input)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if is_write_video:
        print("!!! TYPE:", type(video_output), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(video_output, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        return_value, frame = vid.read()
        if frame is None:
            break
        #frame.show()
        image_array, car_plate_result = identity_car_plate(frame)
        result = image_array
        #image = Image.fromarray(frame)
        #image = yolo.detect_image(image, car_identity)
        #result = np.asarray(image)

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
        if is_write_video:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return None

if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--input", nargs='?', type=str, required=True, help = "Image input path")
    #parser.add_argument("--output", nargs='?', type=str, required=False, default="", help="Image output path")
    FLAGS = parser.parse_args()
    input_image = FLAGS.input
    if not os.path.exists(input_image):
        print(f"Image '{input_image}' is not exist!")
    else:
        print(f"Input image: {input_image}")
        image_processed = process_image_file(input_image)
        image_processed.save("JTDN_CP01_01.jpg", "jpeg")
    #FLAGS.output = "F:/car_plate_ident.mp4"
    #process_video('F:/V00411-144331.mp4', FLAGS.output)