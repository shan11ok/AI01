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
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
warnings.filterwarnings('ignore')

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def main(video_path, output_path=""):

   # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0



   # deep_sort 
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

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
    #yolo = YOLO((416,416))

    line = [(0, h//2), (w, h//2)]

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
    list_file = open('detection.txt', 'w')
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
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        
        # Call the tracker
        tracker.predict()
        counter = tracker.update(detections, line)
        
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlbr()
            p0 = track.last_center()
            p1 = track.center()
            cv2.line(frame, p0, p1, (0,0,255), 3)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            cv2.putText(frame, '%s_%d'%(track.pre_class,track.track_id),(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0),1)

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)

	    # draw line
        cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

	    # draw counter
        start_x = 30
        start_y = 30
        for pre_class in counter:
            cv2.putText(frame, '%s:%d'%(pre_class,counter[pre_class]), (start_x, start_y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            start_y += 20

        cv2.imshow('', frame)
        
        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')
            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
            
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        print("fps= %f"%(fps))
        
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
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
    FLAGS = parser.parse_args()

    main(FLAGS.input, FLAGS.output)
