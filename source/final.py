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
import random
from matplotlib import pyplot as plt
import tools
import aggmodel
import warp

#sanity check
from stn import spatial_transformer_network as transformer
from utils import img2array, array2img
from utils import deg2rad
from PIL import Image

#################################################################################################################################################
#################################################################################################################################################
#DEFINE OUR LOGGER

tf.logging.set_verbosity(tf.logging.INFO)

#PREDIFINED CONSTRAINTS
EPOCH=5
TIME_STEP=4
LR = 0.0001
DEPTH = 2
HEIGHT = 436
WIDTH = 1024
DATA_DIR = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow/alley_2/"
TEST_DIR1 = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow/alley_1/"
TEST_DIR2 = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow/cave_4/"
LOG_DIR = "/media/cvlab/DATA/masa_log/final"
TEST_DIR = './data/alley2/'
ALL_DIR = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow"

##########################################################################################################
#MODEL DEFINITION
##########################################################################################################
#OUR GLOBAL STEP

def normalize(x):
    return tf.divide(tf.subtract(x,tf.reduce_min(x)),tf.subtract(tf.reduce_max(x),tf.reduce_min(x)))

def agg_loss(pred,y):
    #LOSS FOR THE BIRD MODEL
    diff = tf.reduce_sum(tf.square(tf.subtract(y,pred)),axis=-1)
    loss = tf.reduce_mean(diff)
    loss_summary = tf.summary.scalar('agg_loss',loss)

    #RNN OPTIMIZER
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(loss)

    return optimizer,loss_summary

def warp_loss(pred,y):
    #LOSS FOR THE WARP MODEL
    diff = tf.reduce_sum(tf.square(tf.subtract(y,pred)),axis=-1)
    warp_loss = tf.reduce_mean(diff)
    warp_loss_summary = tf.summary.scalar('warp_loss',warp_loss)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(warp_loss)

    return optimizer,warp_loss_summary

x = tf.placeholder(tf.float32,shape=[None,TIME_STEP,2])
x = normalize(x)
y = tf.placeholder(tf.float32,shape=[None,2])
y = normalize(y)

agg_flow = aggmodel.bird_model(x)
opt1,summary1 = agg_loss(agg_flow,y)

agg_flow = tf.concat([x[:,-1,:], agg_flow],axis=-1)
agg_flow = tf.reshape(agg_flow,[1,HEIGHT,WIDTH,4])
#gt = tf.reshape(y,[HEIGHT,WIDTH,DEPTH])
#warped,variables = warp.cnn(agg_flow)
#opt2,summary2 = warp_loss(warped,gt)

saver = tf.train.Saver()
merge1 = tf.summary.merge([summary1])
#merge2 = tf.summary.merge([summary2])

#MODEL END
##########################################################################################################
# define our execution module
##########################################################################################################
def test(test_dir=DATA_DIR,ckpt_path=''):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #my training data as file names
        training_list = [os.path.join(test_dir,file_name) for file_name in os.listdir(test_dir)]
        training_list.sort()
        training_list.reverse()
        train_x,train_y = aggmodel.getTBatch(training_list)

        #tools.showVec(vecs=[train_x[:,:,i,:] for i in range(4)])
        h,w,t,d = train_x.shape

        #predict future trajectories
        agg = sess.run(agg_flow,feed_dict={x: train_x.reshape((h * w,t,d)), y: train_y.reshape((h*w,d))})

        agg = agg.reshape((h,w,d *2))
        warp = agg[:,:,2:]

        print("lossmap bounds: ",np.amin(warp),np.amax(warp))
        projection2 = tools.customWarp(train_x[:,:,-1,:],warp)

        vec1 = tools.readflofile(training_list[1])
        tools.showVec(vecs=[warp,projection2,train_x[:,:,-1,:],train_y])

def train_aggmodel(ckpt_path=''):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #restore model to ckpt if we have one
        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #training/testing logger
        if not os.path.isdir(LOG_DIR):
            os.mkdir(LOG_DIR)
        tools.clearFile(LOG_DIR)

        #FOR LOGGING OUR TRAINING
        training_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'training'),sess.graph)

        #my training data as file names
        training_list = []
        for flow_dir in os.listdir(ALL_DIR):
            training_list.append([os.path.join(ALL_DIR,flow_dir,flow_path) for flow_path in os.listdir(os.path.join(ALL_DIR,flow_dir))])

        #train our model
        counter = 0
        for i in range(EPOCH):
            for j in range(100):
                for s in range(0,50):
                    #PREDICT ONE TIME STEP AHEAD FOR A RANDOM SCENE
                    idx = random.randint(0,len(training_list) - 1)

                    batch = training_list[idx]
                    step_id = random.randint(0,len(batch) - (TIME_STEP + 2))
                    data = batch[step_id:step_id + (TIME_STEP + 1)]

                    print(idx, step_id, step_id + (TIME_STEP + 1))

                    batch.sort()
                    batch.reverse()
                    train_x, train_y = aggmodel.getTBatch(data)
                    h,w,t,d = train_x.shape

                    summary = sess.run([opt1,merge1],feed_dict={x: train_x.reshape((h*w,t,d)), y: train_y.reshape((h*w,d))})
                    training_writer.add_summary(summary[-1],global_step=counter)
                    counter += 1
                    saver.save(sess,'model/final.ckpt')

                print('epoch: %i batch: %i ' % (i+1,j+1))

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# MAIN FUNCTION
if __name__ == '__main__':
    if '-train' in sys.argv:

        arg = sys.argv[sys.argv.index('-train') + 1]
        if arg == 'aggmodel':
            train_aggmodel()
        elif arg == 'both':
            train_aggmodel()
            train_warpmodel()

    elif '-test' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        else:
            print('YOU NEED A MODEL TO TEST ON...')

        test(sys.argv[sys.argv.index('-test') + 1],ckpt_path=model)

    else:
        print('NOT A VALID MODE. EXPECTING [ -train | -test ]')



##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################



