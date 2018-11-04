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
from matplotlib import pyplot as plt


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
EPOCH=3
BATCH_SIZE=20
BATCH_STEP=100
LEARNING_RATE = 0.01
HEIGHT = 436
WIDTH = 1024
DEPTH = 3
T_DIR = './data/train/'
V_DIR = './data/test/'
#data_dir = './data/alley2/'

##########################################################################################################
#MODEL DEFINITION
##########################################################################################################
#OUR GLOBAL STEP
global_step = tf.Variable(0, name='global_step', trainable=False)
inc = tf.assign_add(global_step,1)

with tf.variable_scope('spatial_transformer'):
    with tf.variable_scope('conv'):

        x = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,DEPTH])
        y = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,DEPTH])

        '''
        weights['w_treematter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
                biases['b_treematter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
                tree_conv1 = tf.nn.conv2d(x,weights['w_treematter'],strides=[1,1,1,1],padding='SAME',name='tree_tree1')
        tree1 = tf.nn.relu(tree_conv1 + biases['b_treematter'])
        '''

    with tf.variable_scope('warp'):
        theta = np.array([[1., 0, 0], [1, 1., 1]])
        theta = theta.astype('float32')
        theta = theta.flatten()

        # define loc net weight and bias
        loc_in = HEIGHT * WIDTH * DEPTH
        loc_out = 6
        w_loc = tf.Variable(tf.zeros([loc_in, loc_out]), name='W_loc')
        weight_hist = tf.summary.histogram('weights', w_loc)
        b_loc = tf.Variable(initial_value=theta, name='b_loc')
        bias_hist = tf.summary.histogram('bias', b_loc)

        # tie everything together
        fc_loc = tf.matmul(tf.zeros([BATCH_SIZE, loc_in]), w_loc) + b_loc
        fc_loc_test = tf.matmul(tf.zeros([1, loc_in]), w_loc) + b_loc
        h_trans = transformer(x, fc_loc)
        h_trans_test = transformer(x,fc_loc_test)

    with tf.variable_scope('loss'):
        loss = tf.nn.l2_loss(tf.subtract(y,h_trans)) / (BATCH_SIZE * HEIGHT * WIDTH)
        loss_sum = tf.summary.scalar('loss',loss)
    with tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

saver = tf.train.Saver()

#MODEL END
##########################################################################################################
# define our execution module
##########################################################################################################

def test(img_path, ckpt_path=''):
    input_img = np.zeros((1,HEIGHT,WIDTH,DEPTH),np.float32)
    input_img[0] = cv2.imread(img_path,cv2.IMREAD_COLOR) / float(255)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if ckpt_path != '':
        saver.restore(sess,ckpt_path)

    y = sess.run(h_trans_test, feed_dict={x: input_img})
    y = y * 255
    cv2.imshow('%s warped' % img_path,y[0].astype(np.uint8))
    cv2.waitKey(0)

##########################################################################################################
# define our training module

#GET OUR TESTING BATCH
def getTBatch(flist_x, flist_y):
    #input_img = np.concatenate([img1, img2, img3, img4], axis=0)
    #B, H, W, C = input_img.shape
    #print("Input Img Shape: {}".format(input_img.shape))
    train_x = np.zeros((len(flist_x),HEIGHT,WIDTH,DEPTH),np.float32)
    train_y = np.zeros((len(flist_y),HEIGHT,WIDTH,DEPTH),np.float32)
    for i,f in enumerate(flist_x):
        train_x[i] = cv2.imread(f,cv2.IMREAD_COLOR) / float(255)
    for i,f in enumerate(flist_y):
        train_y[i] = cv2.imread(f,cv2.IMREAD_COLOR) / float(255)

    return train_x, train_y

#GET OUR VALIDATION BATCH
def getVBatch(flist_x,flist_y):
    train_x = np.zeros((len(flist_x),HEIGHT,WIDTH,DEPTH),np.float32)
    train_y = np.zeros((len(flist_y),HEIGHT,WIDTH,DEPTH),np.float32)
    for i,f in enumerate(flist_x):
        train_x[i] = cv2.imread(f,cv2.IMREAD_COLOR) / 255
    for i,f in enumerate(flist_y):
        train_y[i] = cv2.imread(f,cv2.IMREAD_COLOR) / 255

    return train_x, train_y

#training / testing cross validation
def train(ckpt_path=''):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #restore model to ckpt if we have one
        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #training/testing logger
        training_writer = tf.summary.FileWriter('./log/training',sess.graph)
        testing_writer = tf.summary.FileWriter('./log/testing',sess.graph)

        #train our model
        beg = 0
        for i in range(EPOCH):
            #get our entire list of files
            '''
            Vfiles = [os.path.join(V_DIR,f) for f in os.listdir(V_DIR)].sort()
            Tfiles = [os.path.join(T_DIR,f) for f in os.listdir(T_DIR)].sort()

            vbatch = Vfiles[beg:BATCH_SIZE - 1]
            tbatch = Tfiles[beg:BATCH_SIZE - 1]
            beg = batch_size - 1
            '''

            #for testing our warp layer
            testf = os.path.join(T_DIR,os.listdir(T_DIR)[0])
            valf = os.path.join(V_DIR,os.listdir(V_DIR)[0])
            Tfiles = [testf for j in range(BATCH_SIZE)]
            Vfiles = [valf for j in range(BATCH_SIZE)]

            #go through our batch
            for b in range(BATCH_STEP):
                #get the data
                train_x,train_y = getTBatch(Tfiles,Vfiles)
                #valid_x,valid_y = getVBatch(Tfiles,Tfiles)

                #TRAIN THE MODEL
                #merged = tf.summary.merge([loss_sum,weight_hist,bias_hist])
                #summary = sess.run([merged,loss,optimizer],feed_dict={x:train_x,y:train_y})
                summary = sess.run([loss,optimizer],feed_dict={x:train_x,y:train_y})
                del train_x
                del train_y

                #RECORD SUMMARY ON TENSORFLOW AND TERMINAL
                sess.run(inc)
                if b == 0:
                    curr_loss = float(summary[0])
                #training_writer.add_summary(summary[0],global_step=global_step.eval())
                print('epoch: %i  batch_step: %i of %i loss: %.4f \n' % (i+1,b,BATCH_STEP,summary[0]))

                #SAVE THE BEST MODEL
                if(float(summary[0]) <= curr_loss):
                    curr_loss = float(summary[0])
                    saver.save(sess,'model/warp_model.ckpt')
                    print("LOWER LOSS FOUND! model saved")

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
# MAIN FUNCTION

if __name__ == '__main__':
    if '-train' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]

        train(ckpt_path=model)

    if '-test' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]

        test(sys.argv[sys.argv.index('-test') + 1],ckpt_path=model)
    else:
        print('hello world')






