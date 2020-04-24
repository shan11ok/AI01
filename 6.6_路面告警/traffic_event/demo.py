#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import argparse
import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

#0：黑；1：红；2：绿；3：黄；4：蓝；5：洋红；6：青；7：白
BGR_COLOR = [(0,0,0), (0,0,255), (0,255,0), (0,255,255), (255,0,0), (255,0,255), (255,255,0), (255,255,255)]
RGB_COLOR = [(0,0,0), (255,0,0), (0,255,0), (255,255,0), (0,0,255), (255,0,255), (0,255,255), (255,255,255)]

#cv2.putText(frame, 'frame:%d'%(frame_index), (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR[font_color], 2)
'''def draw_text(frame, text, position, font_file, size, color):
    cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype(font_file, size, encoding="utf-8")
    draw.text(position, text, color, font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)'''

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def main(video_path, output_path="", linepos=0.5, direction=2, accl=0., stat_color=7, label_color=7, max_age=7):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 0.7

   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, direction, accl)

    writeVideo_flag = True 
    
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC    = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    video_fps       = video_capture.get(cv2.CAP_PROP_FPS)
    video_size      = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w -= w % 32
    h -= h % 32
    yolo = YOLO((w,h))
    RT = w/700
    #yolo = YOLO((416,416))

    line = [(0, int(h*linepos)), (w, int(h*linepos))]

    isOutput = True if output_path != "" else False
    if isOutput:
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    #if writeVideo_flag:
    # Define the codec and create VideoWriter object
    #    w = int(video_capture.get(3))
    #    h = int(video_capture.get(4))
    #    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #    out = cv2.VideoWriter('output.avi', fourcc, 15, (w, h))
    #list_file = open('detection.txt', 'w')
    frame_index = -1 
        
    fps = 0.0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,pre_classes,scores = yolo.detect_image(image)
       # print("box_num",len(boxs))
        features = encoder(frame,boxs)
        
        # score to 1.0 here).
        detections = [Detection(bbox, pre_class, score, feature) for bbox, pre_class, score, feature in zip(boxs, pre_classes, scores, features)]
        
        # Run non-maxima suppression.
        boxes_np = np.array([d.tlwh for d in detections])
        scores_np = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes_np, nms_max_overlap, scores_np)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        counter = tracker.update(detections, line)

        cv2img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(cv2img)
        draw = ImageDraw.Draw(pilimg)
        font_file = "./Font/ARKai_C.ttf"
        big_font_size = 35
        small_font_size = 30
        stat_font = ImageFont.truetype(font_file, int(big_font_size*RT), encoding="utf-8")
        label_font = ImageFont.truetype(font_file, int(small_font_size*RT), encoding="utf-8")

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1 or track.incident_typ == 0:
                continue 
            bbox = track.to_tlbr()
            p0 = track.last_center()
            p1 = track.center()
            draw.line([p0,p1],RGB_COLOR[label_color],3)
            draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))],None, RGB_COLOR[label_color], 2)
            draw.text((int(bbox[0]-small_font_size*RT*3)+2, int(bbox[1]-small_font_size*RT)+2), '%s_%d'%(track.pre_class,track.track_id), RGB_COLOR[0], font=label_font)
            draw.text((int(bbox[0]-small_font_size*RT*3), int(bbox[1]-small_font_size*RT)), '%s_%d'%(track.pre_class,track.track_id), RGB_COLOR[label_color], font=label_font)
            #cv2.line(frame, p0, p1, (0,0,255), 3)
            #cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            #frame = draw_text(frame, '%s_%d'%(track.pre_class,track.track_id), (int(bbox[0]), int(bbox[1])-20), "./Font/zhcn.ttf", 20, COLOR[font_color])
            #cv2.putText(frame, '%s_%d'%(track.pre_class,track.track_id),(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0),1)
            #cv2.putText(frame, '%s_%d'%(vel_str,track.track_id),(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0),1)

        '''for det in detections:
            bbox = det.to_tlbr()
            #cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))],None, RGB_COLOR[4], 2)'''

	    # draw line
        #cv2.line(frame, line[0], line[1], (0, 255, 255), 5)
        #draw.line([line[0], line[1]],RGB_COLOR[3],5)

	    # draw counter
        frame_index = frame_index + 1
        start_x = 30
        start_y = 120
        #cv2.putText(frame, 'frame:%d'%(frame_index), (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR[font_color], 2)
        #def draw_text(frame, text, position, font_file, size, color):
        #frame = draw_text(frame, '帧序:%d'%(frame_index), (start_x, start_y), "./Font/zhcn.ttf", 20, COLOR[font_color])
        draw.text((start_x+2, start_y+2), '帧序:%d'%(frame_index), RGB_COLOR[0], font=stat_font)
        draw.text((start_x, start_y), '帧序:%d'%(frame_index), RGB_COLOR[stat_color], font=stat_font)
        start_y += big_font_size*RT
        track_type_list = sorted(counter.keys())
        track_type_list.reverse()
        for track_type in track_type_list:
            #frame = draw_text(frame, '%s:%d'%(track_type,counter[track_type]), (start_x, start_y), "./Font/zhcn.ttf", 20, COLOR[font_color])
            #cv2.putText(frame, '%s:%d'%(pre_class,counter[pre_class]), (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR[font_color], 2)
            draw.text((start_x+2, start_y+2), '%s:%d'%(track_type,counter[track_type]), RGB_COLOR[0], font=stat_font)
            draw.text((start_x, start_y), '%s:%d'%(track_type,counter[track_type]), RGB_COLOR[stat_color], font=stat_font)
            start_y += big_font_size*RT
        
        frame = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            '''list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
                    list_file.write('%s %s '%(pre_classes[i],scores[i]))
                pick_str = ' '.join(['%d'%(x) for x in indices])
                list_file.write('Pick num is: %s'%(pick_str))
            list_file.write('\n')'''
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        #list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    parser.add_argument(
        "--linepos", nargs='?', type=float, default="0.5",
        help = "[Optional] reference line position"
    )

    parser.add_argument(
        "--direction", nargs='?', type=int, default="2",
        help = "[Optional] direction, 1: one direction, 2: dual direction"
    )

    parser.add_argument(
        "--accl", nargs='?', type=float, default="0.",
        help = "[Optional] car acceleration"
    )

    parser.add_argument(
        "--stat_color", nargs='?', type=int, default="7",
        help = "[Optional] statistic data color"
    )

    parser.add_argument(
        "--label_color", nargs='?', type=int, default="7",
        help = "[Optional] label color"
    )

    parser.add_argument(
        "--maxage", nargs='?', type=int, default="7",
        help = "[Optional] maxage"
    )

    FLAGS = parser.parse_args()

    main(FLAGS.input, FLAGS.output, FLAGS.linepos, FLAGS.direction, FLAGS.accl, FLAGS.stat_color, FLAGS.label_color, FLAGS.maxage)
