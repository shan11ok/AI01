#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
基于迁移学习进行图像分类模型训练
'''

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad
from myutil.Util import Util

# 定义必要变量
############################
# 模型训练时的Batch Size
batch_size = 16
# 训练数据目录
train_data_dir = f"F:/data"
# 验证集数据目录
validation_data_dir = f"F:/data"
# 训练样本数据量
train_sample_num = Util.count_file_in_dir(train_data_dir)
# 训练验证集数据量
validation_sample_num = 60
# 目标分类数量
class_num = Util.count_subdir_in_dir(train_data_dir)
# 模型保存名称
model_file = "vehicle_iden_model"


# 训练数据准备
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

# 验证数据准备
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=(299, 299),#Inception V3规定大小
                                                    batch_size=batch_size)

val_generator = val_datagen.flow_from_directory(directory=validation_data_dir,
                                                target_size=(299, 299),
                                                batch_size=batch_size)

# 构建基础模型
base_model = InceptionV3(weights='imagenet', include_top=False)
# 增加新的输出层
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024, activation='relu')(x)
predictions = Dense(class_num, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


def setup_to_transfer_learning(model,base_model):#base_model
    '''
    这里的base_model和model里面的iv3都指向同一个地址
    '''
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 17 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

print(train_generator.class_indices)
Util.save_dict(f"dict.txt", train_generator.class_indices)

# 迁移学习
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=train_sample_num // batch_size,
                    epochs=20,
                    validation_data=val_generator,
                    validation_steps=12,#12
                    class_weight='auto'
                    )

model.save(f"./models/{model_file}_iv3_tl.h5")

# 训练全连接层
setup_to_fine_tune(model, base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=validation_sample_num // batch_size,
                                 epochs=30,
                                 validation_data=val_generator,
                                 validation_steps=12,
                                 class_weight='auto')
model.save(f"./models/{model_file}_iv3_ft.h5")

print(train_generator.class_indices)
