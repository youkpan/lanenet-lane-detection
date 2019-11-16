#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""

import sys
sys.path.append("D:\\self-driving\\lanenet-lane-detection")

import argparse
import os.path as ops
import time

import cv2
import glog as log
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

CFG = global_config.cfg
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir')
    parser.add_argument('--weights_path', type=str, help='The model weights path')

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr

def frame2numpy(frame, frameSize):
    buff = np.fromstring(frame, dtype='uint8')
    # Scanlines are aligned to 4 bytes in Windows bitmaps
    strideWidth = int((frameSize[0] * 3 + 3) / 4) * 4
    # Return a copy because custom strides are not supported by OpenCV.
    return as_strided(buff, strides=(strideWidth, 3, 1), shape=(frameSize[1], frameSize[0], 3)).copy()

def get_xy(x,y,width,height):
    if x<0:
        x=0
    if x>width-1:
        x=width-1

    if y<0:
        y=0
    if y>height-1:
        y=height-1

    return x,y


def find_target_point(image):
    #image_path = "D:\\self-driving\\lanenet-lane-detection\\data\\source_image\\lanenet_binary_seg.png"
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(image.shape)

    height = image.shape[0]# 720
    width = image.shape[1] #1280
    topy = int(353/720*height)
    bottomy = int(600/720*height)

    '''
    image = np.zeros(shape=(hight, width,3), dtype=np.uint8)
    for i in range(len(lane_mark_array)):
        print(int(lane_mark_array[i][0]),
                        int(lane_mark_array[i][1]))
        cv2.circle(image, (int(lane_mark_array[i][0]),
                        int(lane_mark_array[i][1])), 2, [0, 0, 255], -1)
    '''
    

    image2 = image.copy()
    center_x = int(width/2)
    find_y = 0
    for y in range(bottomy,topy,-2):

        for incx in range(int(width*0.4)):
            find =0
            xx1=center_x-incx

            xx1,yy = get_xy(xx1,y,width,height)
            #print(xx1,y,image[y][xx1])

            if image[y][xx1]>100 :
                #print(xx1,y)
                for dx in range(int(width*0.2)):
                    xx = center_x+incx-int(width*0.1)+dx

                    xx,yy = get_xy(xx,y,width,height)
                    if image[y][xx]>100 :

                        center_x1 = (xx1 + xx)/2
                        #print(center_x1,center_x)

                        if abs(center_x1 - center_x)< int(center_x*0.4):
                            center_x = int(center_x1)
                            find = 1
                            find_y = y
                            #print("find")
                            break

            if find==1:
                #cv2.circle(image2, (int(xx1), int(y)), 1, [255], -1)
                #cv2.circle(image2, (int(xx), int(y)), 1, [255], -1)
                #cv2.circle(image2, (int(center_x), int(y)), 1, [ 255], -1)
                break

    #plt.figure('image')
    #plt.imshow(image2[:, :])
    #plt.show()

    print("target center_x",center_x,find_y)
    return center_x,find_y

class mlanenet:
    def __init__(self,weights_path):

        with tf.device('/cpu:0'):
            self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

            net = lanenet.LaneNet(phase='test', net_flag='vgg')
            self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

            self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

            saver = tf.train.Saver()
            # Set sess configuration
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
            sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
            sess_config.gpu_options.allocator_type = 'BFC'
            self.sess = tf.Session(config=sess_config)
            with self.sess.as_default():
                saver.restore(sess=self.sess, save_path=weights_path)

    def inference(self,image):
        with tf.device('/cpu:0'):
            image_vis = image
            image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)

            image = image / 127.5 - 1.0
            
            #log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

            with self.sess.as_default():

                t_start = time.time()
                binary_seg_image, instance_seg_image = self.sess.run(
                    [self.binary_seg_ret, self.instance_seg_ret],
                    feed_dict={self.input_tensor: [image]}
                )

                x,y = find_target_point(binary_seg_image[0]* 255)

                t_cost = time.time() - t_start
                log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

                postprocess_result = self.postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis,
                    #data_source='GTAV640x320'
                )
                mask_image = postprocess_result['mask_image']
                lane_mark_array = []
                try:
                    lane_mark_array = postprocess_result['lane_mark_array']
                except Exception as e:
                    pass
                #print(len(lane_mark_array))

                for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
                embedding_image = np.array(instance_seg_image[0], np.uint8)

                '''
                plt.figure('mask_image')
                plt.imshow(mask_image[:, :, (2, 1, 0)])
                plt.figure('src_image')
                plt.imshow(image_vis[:, :, (2, 1, 0)])
                plt.figure('instance_image')
                plt.imshow(embedding_image[:, :, (2, 1, 0)])
                plt.figure('binary_image')
                plt.imshow(binary_seg_image[0] * 255, cmap='gray')
                plt.show()
                '''

                return x,y,postprocess_result['source_image'],lane_mark_array,binary_seg_image[0] * 255


def test_lanenet(image, weights_path):

    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0

    #log.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', net_flag='vgg')
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor()

    saver = tf.train.Saver()

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run(
            [binary_seg_ret, instance_seg_ret],
            feed_dict={input_tensor: [image]}
        )

        x,y = find_target_point(binary_seg_image[0]* 255)

        t_cost = time.time() - t_start
        log.info('Single imgae inference cost time: {:.5f}s'.format(t_cost))

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=image_vis,
            data_source='GTAV640x320'
        )
        mask_image = postprocess_result['mask_image']
        lane_mark_array = []
        try:
            lane_mark_array = postprocess_result['lane_mark_array']
        except Exception as e:
            pass
        
        #print("len(lane_mark_array)",len(lane_mark_array))

        for i in range(CFG.TRAIN.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (2, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

        cv2.imwrite('instance_mask_image.png', mask_image)
        cv2.imwrite('source_image.png', postprocess_result['source_image'])
        cv2.imwrite('binary_mask_image.png', binary_seg_image[0] * 255)

    sess.close()

    return


if __name__ == '__main__':
    """
    test code
    """
    # init args
    #args = init_args()
    image_path="D:\\self-driving\\lanenet-lane-detection\\data\\training_data_example\\image\\0001.png"
    weights_path="D:\\self-driving\\lanenet-lane-detection\\checkpoint\\tusimple_lanenet_vgg.ckpt"
    #test_lanenet( image_path,  weights_path)
    #l= mlanenet(weights_path)
     
    #image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #l.inference(image)



    #a=np.array([[571.7042971545504, 251.44452164020944], [542.0828992451794, 259.85516876870014], [507.02466509601925, 271.01783005734706], [472.655441991815, 280.7059735237284], [428.5728722831689, 293.28556596471907], [394.55882778587886, 302.5377353100066], [352.33342818224287, 314.17985826857546], [327.855543557948, 320.20927981112874], [303.83966913548005, 327.50917864860367], [229.08879498256948, 347.01253817943814], [227.63411282685271, 347.45113940949136], [177.81894631293943, 360.6762091454039], [118.70541513327774, 376.46554890084764], [103.47403889325442, 380.35901463285404], [82.1009561113544, 385.3812645445479]])

    
