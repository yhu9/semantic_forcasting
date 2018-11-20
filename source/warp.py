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

#################################################################################################################################################
#################################################################################################################################################
#DEFINE OUR LOGGER

tf.logging.set_verbosity(tf.logging.INFO)

#PREDIFINED CONSTRAINTS
EPOCH=10
BATCH_STEP=100
LEARNING_RATE = 0.01
HEIGHT = 436
WIDTH = 1024
DEPTH = 3
T_DIR = '/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/clean/sleeping_2/'
LOG_DIR = "/media/cvlab/DATA/masa_log/warp"
#data_dir = './data/alley2/'

##########################################################################################################
#MODEL DEFINITION
##########################################################################################################
#OUR GLOBAL STEP


def cnn(input_):
    w1 = tf.Variable(tf.random_normal([13,13,4,32],dtype=tf.float32),dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal([7,7,32,10],dtype=tf.float32),dtype=tf.float32)
    w3 = tf.Variable(tf.random_normal([3,3,10,2],dtype=tf.float32),dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal([32],dtype=tf.float32),dtype=tf.float32)
    b2 = tf.Variable(tf.random_normal([10],dtype=tf.float32),dtype=tf.float32)
    b3 = tf.Variable(tf.random_normal([2],dtype=tf.float32),dtype=tf.float32)

    conv1 = tf.nn.conv2d(input_,w1,strides=[1,1,1,1],padding='SAME')
    conv1_out = tf.nn.sigmoid(tf.nn.bias_add(conv1,b1))
    conv2 = tf.nn.conv2d(conv1_out,w2,strides=[1,1,1,1],padding='SAME')
    conv2_out = tf.nn.sigmoid(tf.nn.bias_add(conv2,b2))
    conv3 = tf.nn.conv2d(conv2_out,w3,strides=[1,1,1,1],padding='SAME')
    conv3_out = tf.nn.bias_add(conv3,b3)

    return conv3_out,[w1,w2,w3,b1,b2,b3]

def warp(input_x):
	with tf.variable_scope('spatial_transformer'):
		with tf.variable_scope('conv'):
			'''
			weights['w_treematter'] = tf.Variable(tf.random_normal([7,7,3,constants.CNN_LOCAL1]))
					biases['b_treematter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
					tree_conv1 = tf.nn.conv2d(x,weights['w_treematter'],strides=[1,1,1,1],padding='SAME',name='tree_tree1')
			tree1 = tf.nn.relu(tree_conv1 + biases['b_treematter'])
			'''
        with tf.variable_scope('warp'):
            theta = np.array([[1., 0, 0], [0, 1., 0]])
            theta = theta.astype('float32')
            theta = theta.flatten()

			# define loc net weight and bias
            loc_in = HEIGHT * WIDTH * DEPTH
            loc_out = 6
            w_loc = tf.Variable(tf.zeros([loc_in, loc_out]), name='W_loc')
            b_loc = tf.Variable(initial_value=theta, name='b_loc')

			# tie everything together
            # note we have to have 2 different outputs here because tensorflow expects a defined value for the batch size
            # and for testing we have batch size of 1, but for training we have larger batch sizes

            fc_loc = tf.matmul(tf.zeros([1, loc_in]), w_loc) + b_loc
            h_trans = transformer(input_x, fc_loc)

        return h_trans

def warp_model(mode='train'):

    x = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,DEPTH])
    y = tf.placeholder(tf.float32,[None,HEIGHT,WIDTH,DEPTH])

    out= warp(x)

    loss = tf.nn.l2_loss(tf.subtract(y,out)) / (1 * HEIGHT * WIDTH)
    loss_sum = tf.summary.scalar('loss',loss)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    merged = tf.summary.merge([loss_sum])

    saver = tf.train.Saver()
    return x,y,saver,[out,loss,optimizer,merged]

#MODEL END
##########################################################################################################
# define our execution module
##########################################################################################################
#GET OUR TESTING BATCH
def getTBatch(flist_x):
    #input_img = np.concatenate([img1, img2, img3, img4], axis=0)
    #B, H, W, C = input_img.shape
    #print("Input Img Shape: {}".format(input_img.shape))
    if len(flist_x) % 2 == 1:
        flist_x.pop()

    train_x = np.zeros((int(len(flist_x)/2),HEIGHT,WIDTH,DEPTH),np.float32)
    train_y = np.zeros((int(len(flist_x)/2),HEIGHT,WIDTH,DEPTH),np.float32)

    counter = 0
    for i,f in enumerate(flist_x):
        j = i + 1
        img = cv2.imread(f,cv2.IMREAD_COLOR)
        if j % 2 == 1:
            train_x[counter] = img
        else:
            train_y[counter] = img
            counter += 1

    return train_x,train_y

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

#training / testing cross validation
def train(ckpt_path=''):
    x,y,saver,params = warp_model()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        #restore model to ckpt if we have one
        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #training/testing logger
        training_writer = tf.summary.FileWriter(os.path.join(LOG_DIR,'training'),sess.graph)

        #Training files
        Tfiles = [os.path.join(T_DIR, f_in) for f_in in os.listdir(T_DIR)]
        Tfiles.sort()

        #train our model
        counter = 0
        for i in range(EPOCH):
            #for testing our warp layer
            #go through our batch
            train_x,train_y = getTBatch(Tfiles)

            for x_in,y_in in zip(train_x,train_y):
                #TRAIN THE MODEL

                x_in = x_in.reshape((1,x_in.shape[0],x_in.shape[1],x_in.shape[2]))
                y_in = y_in.reshape((1,y_in.shape[0],y_in.shape[1],y_in.shape[2]))
                summary = sess.run(params,feed_dict={x:x_in,y:y_in})
                training_writer.add_summary(summary[-1],global_step=counter)

                #RECORD SUMMARY ON TENSORFLOW AND TERMINAL
                #SAVE THE BEST MODEL
                saver.save(sess,'model/warp_model.ckpt')
                counter +=1

            print('epoch: %i  loss: %.4f' % (i+1,summary[1]))


##########################################################################################################
# define our training module
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

    elif '-test' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]

        test(sys.argv[sys.argv.index('-test') + 1],ckpt_path=model)
    else:
        print('hello world')



