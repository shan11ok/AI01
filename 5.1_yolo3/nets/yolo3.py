from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from nets.darknet53 import darknet_body
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积(包括正则化/标准化/leaky激活函数)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    #向上采样
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1))(x)

    # 向右输出，经过3*3的卷积和将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)
            
    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(inputs, num_anchors, num_classes):
    # 生成darknet53的主干模型
    #52,52,256
    #26,26,512
    #13,13,1024
    feat1,feat2,feat3 = darknet_body(inputs)
    darknet = Model(inputs, feat3)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85) 85是用了COCO数据集，25用的是VOC数据集
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat2])
    # 第二个特征层
    # y2=(batch_size,26,26,3,85)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,feat1])
    # 第三个特征层
    # y3=(batch_size,52,52,3,85)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

#---------------------------------------------------#
#   将预测值的每个特征层调成真实值【解码的核心】
#---------------------------------------------------#
def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    # 1，13，13，3，85
    num_anchors = len(anchors)
    # 对先验框进行reshape才能和特征层进行运算，[1, 1, 1, num_anchors, 2]
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    # 获得x，y的网格
    # (13, 13, 1, 2)，建议一个13*13的网格
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    #里面的内容类似[[0,0],[0,1],[0,2].......[1,0]]用列表的方式对应网格的点

    # 对特征层进行reshape成(batch_size,13,13,3,85)
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 将预测值调成真实值
    # box_xy对应框的中心点，将特征层的前两个参数x_offset和y_offset，加上x，y的偏移再除以13
    # box_wh对应框的宽和高，取指数再乘先验框/(416*416)
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    # 在计算loss的时候返回如下参数
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   对box进行调整，使其符合真实图片的样子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    #把yolo3预先输入的包含上下灰边的图片，去掉灰边还原比例
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    #转换数据类型float32进行指数运算
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    #根据xy的偏移量获取框的左上角的坐标和右下角的坐标，方便后续画图，中心点+长宽/2
    #返回boxes包含框的四个点的位置
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   获取每个box的位置和它的得分
#---------------------------------------------------#
def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    # yolo_head将预测值调成真实值
    # box_xy对应框的中心点，xy轴偏移量
    # box_wh对应框的宽和高，框的长和宽
    # box_confidence对应框的置信度
    # box_class_probs对应框的预测结果的概率
    # -1,13,13,3,2; -1,13,13,3,2; -1,13,13,3,1; -1,13,13,3,80
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # yolo_correct_boxes将box_xy、和box_wh调节成y_min,y_max,xmin,xmax
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    # 获得得分和box

    # 每个点有4个参数把boxes进行平铺，[y_min,y_max,xmin,xmax]....
    boxes = K.reshape(boxes, [-1, 4])

    # 得分=置信度*每个类的概率
    box_scores = box_confidence * box_class_probs
    # 把得分平铺，输出每个boxes框80个类的可能性
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

#---------------------------------------------------#
#   图片预测
#---------------------------------------------------#
def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # 获得特征层的数量
    num_layers = len(yolo_outputs)
    # 特征层1对应的anchor是678
    # 特征层2对应的anchor是345
    # 特征层3对应的anchor是012
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]]
    
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    #426*426
    boxes = []
    box_scores = []
    # 对每个特征层进行处理yolo_boxes_and_scores获取boxes位置和得分
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 将每个特征层的结果进行堆叠
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)

    #判断位置是否大于门线，相当于框里面有东西有类别
    # [n,80]
    # box_scores[1,1]，取第2个框中第2类的概率
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # 取出所有box_scores >= score_threshold的框，和成绩
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        # 非极大抑制，去掉box重合程度高的那一些，取得分高的框
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)

        # 获取非极大抑制后的结果
        # 下列三个分别是
        # 框的位置，得分与种类
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


