#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

#################################################################################################################################################
#################################################################################################################################################
#IMPORTS

import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import tools
import random
import shutil
from matplotlib import pyplot as plt

#################################################################################################################################################
#################################################################################################################################################

#IF YOU WANT TO SET THE SESSIONS TO RUN ON CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#DEFINE OUR LOGGER
tf.logging.set_verbosity(tf.logging.INFO)
#PREDIFINED CONSTRAINTS
EPOCH=10
BATCH_STEP=1000
LEARNING_RATE = 0.0001
HEIGHT = 436
WIDTH = 1024
DEPTH = 2
PATCH_HEIGHT = HEIGHT
PATCH_WIDTH = WIDTH
T_DIR1 = './data/mountain_1'
T_DIR2 = './data/sleeping_1'
T_DIR3 = './data/sleeping_2'
V_DIR = './data/sleeping_1'
all_dir = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow"
#################################################################################################################################################
#################################################################################################################################################
#function for tiling an image into patches of size w * h
#########################################################################################################
#MODEL DEFINITION
##########################################################################################################
with tf.variable_scope('seg_predictor'):

    #input are optical flow vectors as patches
    x = tf.placeholder(tf.float32,[None,PATCH_HEIGHT,PATCH_WIDTH,DEPTH])
    #x_reshaped = tf.reshape(x,[-1,PATCH_WIDTH * PATCH_HEIGHT,DEPTH])
    #vecs = tf.split(x_reshaped,2,axis=2)

#saver = tf.train.Saver()
#merged = tf.summary.merge([summ for summ in summs])
#MODEL END
##########################################################################################################
# define our execution module
##########################################################################################################
def test(flow_path, ckpt_path=''):

    if os.path.isdir(flow_path):
        a= os.listdir(flow_path)
        a.sort()
        for file_name in a:
            full_path = os.path.join(flow_path,file_name)
            test(full_path)
        return True

##########################################################################################################
# define our training module
#GET OUR TESTING BATCH
def getTBatch(f):
    flowfield = tools.readflofile(f)

    return flowfield.reshape((1,flowfield.shape[0],flowfield.shape[1],flowfield.shape[2]))

#training / testing cross validation
def train(ckpt_path='',mode='still',name='cam_model_small'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #restore model to ckpt if we have one
        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #training/testing logger
        if not os.path.isdir("./log/cam_model_small"):
            os.mkdir("./log/cam_model_small")

        training_writer = tf.summary.FileWriter('./log/cam_model_small/training',sess.graph)
        testing_writer = tf.summary.FileWriter('./log/cam_model_small/testing',sess.graph)

        #my training data as file names
        training_list = []
        if mode == 'still':
            training_list.append([os.path.join(T_DIR1,fname) for fname in os.listdir(T_DIR1)])
            training_list.append([os.path.join(T_DIR2,fname) for fname in os.listdir(T_DIR2)])
            training_list.append([os.path.join(T_DIR3,fname) for fname in os.listdir(T_DIR3)])
        elif mode == 'all':
            for flow_dir in os.listdir(all_dir):
                training_list += [os.path.join(all_dir,flow_dir,flow_path) for flow_path in os.listdir(os.path.join(all_dir,flow_dir))]

        #our training steps
        counter = 0
        for i in range(EPOCH):
            for j in range(len(training_list)):
                train_x = getTBatch(training_list[j])
                #merge our summaries
                summary = sess.run([cost,opt,merged],feed_dict={x: train_x[:]})
                lossvalue = summary[0]
                curr_loss = float(lossvalue)

                #RECORD SUMMARY ON TENSORFLOW AND TERMINAL
                training_writer.add_summary(summary[-1],global_step=counter)
                counter += 1

                #SAVE THE BEST MODEL
                curr_loss = float(lossvalue)
                saver.save(sess,os.path.join("model",name) + '.ckpt')
            print('epoch: %i batch_size: %i loss: %.4f ' % (i+1,j,lossvalue))

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
# MAIN FUNCTION

if __name__ == '__main__':
    if '-train' in sys.argv:
        model = ''
        mode = 'still'
        name = 'cam_model_small'
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        if '-all' in sys.argv:
            mode = 'all'
        if '-name' in sys.argv:
            name = sys.argv[sys.argv.index('-name') + 1]

        train(ckpt_path=model,mode=mode,name=name)

    elif '-test' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        else:
            print('YOU NEED A MODEL TO TEST ON...')

        test(sys.argv[sys.argv.index('-test') + 1],ckpt_path=model)

    else:
        img=  cv2.imread(sys.argv[1])
        tiled = get_tile_images(img)
        print(tiled.shape)
        print(tiled)

