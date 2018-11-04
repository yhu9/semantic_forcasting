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
import time
import keras

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
LOG_DIR = "/media/cvlab/DATA/masa_log/bird_model"
TEST_DIR = './data/alley2/'
ALL_DIR = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow"

def normalize(x):
    return tf.divide(tf.subtract(x,tf.reduce_min(x)),tf.subtract(tf.reduce_max(x),tf.reduce_min(x)))

##########################################################################################################
#MODEL DEFINITION
##########################################################################################################
#OUR GLOBAL STEP
global_step = tf.Variable(0, name='global_step', trainable=False)
inc = tf.assign_add(global_step,1)

def bird_model1(input_):
    n_hidden = 100

    cell = tf.nn.rnn_cell.LSTMCell(100)

    outputs, states = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32)

    w = tf.Variable(tf.random_normal([n_hidden,2]))
    b = tf.Variable(tf.random_normal([2]))

    return tf.matmul(outputs[:,-1,:],w) + b

#x = tf.placeholder(tf.float32,shape=[TIME_STEP,2])
#y = tf.placeholder(tf.float32,shape=[1,2])
x = tf.placeholder(tf.float32,shape=[None,TIME_STEP,2])
x = normalize(x)
y = tf.placeholder(tf.float32,shape=[None,2])
y = normalize(y)

outputs = bird_model1(x)

diff = tf.reduce_sum(tf.square(tf.subtract(y,outputs)),axis=-1)
loss = tf.reduce_mean(diff)
loss_summary = tf.summary.scalar('loss',loss)

#RNN OPTIMIZER
optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(loss)

#seq_length = tf.placeholder(tf.int32)
saver = tf.train.Saver()
merged = tf.summary.merge([loss_summary])

'''
#OUR RNN FRAMEWORK
# reshape to [1, TIME_STEP]
# (eg. [had] [a] [general] -> [20] [6] [33])
#x = tf.placeholder(tf.float32,shape=(TIME_STEP,DEPTH))
x = tf.placeholder(tf.float32,shape=[None,TIME_STEP,HEIGHT,WIDTH,2])
y = tf.placeholder(tf.float32,shape=[None,HEIGHT,WIDTH,2])

convLSTM_cell = tf.contrib.rnn.ConvLSTMCell(
    conv_ndims=2,
    input_shape = [HEIGHT,WIDTH,DEPTH],
    output_channels=2,
    kernel_shape=[3,3]
    )

outputs, states = tf.nn.dynamic_rnn(convLSTM_cell, x, dtype=tf.float32)

#CONSIDERING USING A CONVOLUTION NETWORK AFTER THE OUTPUT OF CONV LSTM
weights1 =  tf.Variable(tf.random_normal([3,3,2,32]))
biases1 = tf.Variable(tf.random_normal([32]))
conv_out1 = tf.nn.conv2d(outputs[:,-1,:,:,:],weights1,strides=[1,1,1,1],padding='SAME')
out1 = tf.nn.sigmoid(conv_out1 + biases1)

weights2 =  tf.Variable(tf.random_normal([3,3,32,32]))
biases2 = tf.Variable(tf.random_normal([32]))
conv_out2 = tf.nn.conv2d(out1,weights2,strides=[1,1,1,1],padding='SAME')
out2 = tf.nn.sigmoid(conv_out2 + biases2)

weights3 =  tf.Variable(tf.random_normal([3,3,32,2]))
biases3 = tf.Variable(tf.random_normal([2]))
conv_out3 = tf.nn.conv2d(out2,weights3,strides=[1,1,1,1],padding='SAME')
out3 = conv_out3 + biases3

loss = tf.reduce_mean(tf.square(tf.losses.absolute_difference(y[0],outputs[0][-1])))
loss_summary = tf.summary.scalar('loss',loss)

#RNN OPTIMIZER
optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(loss)

#seq_length = tf.placeholder(tf.int32)
saver = tf.train.Saver()
merged = tf.summary.merge([loss_summary])
'''
#MODEL END
##########################################################################################################
# define our execution module
##########################################################################################################

def test1(test_dir=DATA_DIR,ckpt_path=''):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #my training data as file names
        training_list = [os.path.join(test_dir,file_name) for file_name in os.listdir(test_dir)]
        training_list.sort()
        training_list.reverse()
        train_x,train_y = getTBatch(training_list)
        h,w,t,d = train_x.shape

        #predict future trajectories
        out, lossval, lossmap = sess.run([outputs,loss,diff],feed_dict={x: train_x.reshape((h * w,t,d)), y: train_y.reshape((h*w,d))})

        vec = out.reshape((h,w,d))

        print(np.amin(lossmap),np.amax(lossmap))

        vec1 = tools.readflofile(training_list[1])
        tools.showVec(vecs=[vec,train_y])

        lossmap -= np.amin(lossmap)
        lossmap = lossmap / np.amax(lossmap) * 255
        cv2.imshow('lossmap',lossmap.reshape((h,w)).astype(np.uint8))
        cv2.waitKey(0)

##########################################################################################################
# define our training module

def getFakeData():
    BATCH_SIZE=20
    #input_img = np.concatenate([img1, img2, img3, img4], axis=0)
    #B, H, W, C = input_img.shape
    #print("Input Img Shape: {}".format(input_img.shape))

    #TRAIN ON CONSTANT VECTORS
    const_speed_training = np.zeros((BATCH_SIZE,TIME_STEP,DEPTH))
    vector1 = np.linspace(-1,1,BATCH_SIZE)
    vector2 = np.random.rand((BATCH_SIZE))
    const_speed_training[:,0,0] = vector1
    const_speed_training[:,1,0] = vector1
    const_speed_training[:,2,0] = vector1
    const_speed_training[:,3,0] = vector1
    const_speed_training[:,0,1] = vector2
    const_speed_training[:,1,1] = vector2
    const_speed_training[:,2,1] = vector2
    const_speed_training[:,3,1] = vector2

    #TRAIN ON VARYING SPEEDS
    #const_speedx = np.ones((BATCH_SIZE,TIME_STEP,DEPTH))
    #const_speedy = np.ones((BATCH_SIZE,1,2))

    #TRAIN ON VARYING ACCELERATIONS
    #const_speedx = np.ones((BATCH_SIZE,TIME_STEP,DEPTH))
    #const_speedy = n

    return const_speed_training,const_speed_training[:,0,:].reshape(BATCH_SIZE,1,DEPTH)


#GET OUR TESTING BATCH
def getTBatch(files):

    train_x = stitch_flo_files(files[1:])
    train_y = tools.readflofile(files[0])

    return train_x, train_y

def stitch_flo_files(files):
    length = len(files)
    flow = tools.readflofile(files[0])
    h,w,d = flow.shape

    flows = np.zeros((length,h,w,d),dtype=np.float32)
    i = 1
    flows[0] = flow

    for f in files[1:]:

        flow = tools.readflofile(f)
        flows[i] = flow
        i += 1

    return tools.stitchFlow(flows)

'''
#training / testing cross validation
def train(ckpt_path='',mode='real'):
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
        if mode == 'real':
            for flow_dir in os.listdir(ALL_DIR):
                training_list.append([os.path.join(ALL_DIR,flow_dir,flow_path) for flow_path in os.listdir(os.path.join(ALL_DIR,flow_dir))])

        #STORE ENTIRE EPOCH IN MEMORY FOR FASTER TRAINING?

        #train our model
        counter = 0
        for i in range(EPOCH):
            for j,batch in enumerate(training_list):

                #PREDICT ONE TIME STEP AHEAD FOR A PARICULAR SCENE
                for s in range(0,len(batch),TIME_STEP + 1):

                    if s + TIME_STEP >= len(batch):
                        break

                    batch.sort()
                    batch.reverse()
                    train_x, train_y = getTBatch(batch[s:s+TIME_STEP + 1])

                    col = random.randint(0,WIDTH - 1)
                    row = random.randint(0,HEIGHT - 1)

                    summary = sess.run([loss,optimizer,merged],feed_dict={x: [train_x[:,col,row,:], y: train_y[col,row]})
                    lossvalue = summary[0]

                    #RECORD SUMMARY ON TENSORFLOW AND TERMINAL
                    #sess.run(inc)
                    #if s == 0:
                    #    curr_loss = float(lossvalue)

                    training_writer.add_summary(summary[-1],global_step=counter)
                    counter += 1
                    #SAVE THE BEST MODEL
                    #if(float(lossvalue) <= curr_loss):
                    #    curr_loss = float(lossvalue)
                    #   print("LOWER LOSS FOUND! model saved")
                    saver.save(sess,'model/bird_model.ckpt')

                print('epoch: %i batch: %s loss: %.4f \n' % (i+1,batch[0][:-3],lossvalue))
'''

#training / testing cross validation
def train(ckpt_path='',mode='real'):
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
        if mode == 'real':
            for flow_dir in os.listdir(ALL_DIR):
                training_list.append([os.path.join(ALL_DIR,flow_dir,flow_path) for flow_path in os.listdir(os.path.join(ALL_DIR,flow_dir))])


        #train our model
        counter = 0
        for i in range(EPOCH):
            for j in range(100):
                #PREDICT ONE TIME STEP AHEAD FOR A PARICULAR SCENE
                for s in range(0,50):

                    idx = random.randint(0,len(training_list) - 1)
                    batch = training_list[idx]
                    step_id = random.randint(0,len(batch) - (TIME_STEP + 2))
                    data = batch[step_id:step_id + (TIME_STEP + 1)]

                    if s + TIME_STEP >= len(batch):
                        break

                    batch.sort()
                    batch.reverse()
                    train_x, train_y = getTBatch(data)
                    h,w,t,d = train_x.shape

                    summary = sess.run([loss,optimizer,merged],feed_dict={x: train_x.reshape((h*w,t,d)), y: train_y.reshape((h*w,d))})
                    lossvalue = summary[0]

                    #RECORD SUMMARY ON TENSORFLOW AND TERMINAL
                    #sess.run(inc)
                    #if s == 0:
                    #    curr_loss = float(lossvalue)

                    training_writer.add_summary(summary[-1],global_step=counter)
                    counter += 1
                    #SAVE THE BEST MODEL
                    #if(float(lossvalue) <= curr_loss):
                    #    curr_loss = float(lossvalue)
                    #   print("LOWER LOSS FOUND! model saved")
                    saver.save(sess,'model/bird_model.ckpt')

                print('epoch: %i loss: %.4f \n' % (i+1,lossvalue))

#do our inference on the files given
#there should be 4 optical flow files
def predict(files,seq_len=4,ckpt_path=''):

    pass

##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

# MAIN FUNCTION
if __name__ == '__main__':
    if '-train' in sys.argv:
        model = ''
        ex_data = 'real'
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        if '-example' in sys.argv:
            ex_data = sys.argv[sys.argv.index('-example') + 1]

        train(ckpt_path=model,mode=ex_data)
    if '-test1' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        else:
            print('YOU NEED A MODEL TO TEST ON...')

        test1(sys.argv[sys.argv.index('-test1') + 1],ckpt_path=model)

    if '-predict' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        else:
            print('YOU NEED A MODEL TO TEST WITH...')

        files = [os.path.join('data/market2_t4/',f) for f in os.listdir('data/market2_t4/')]
        predict(files,ckpt_path=model)

    else:
        print('hello world')


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

'''
def model():
    n_hidden = 20
    TIME_STEP = 4

    #OUR RNN FRAMEWORK
    # reshape to [1, TIME_STEP]
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.placeholder(tf.float32,shape=(TIME_STEP,DEPTH))
    y = tf.placeholder(tf.float32,shape=(1,DEPTH))
    sequence = tf.reshape(x,[TIME_STEP,DEPTH])
    sequence = tf.split(sequence,TIME_STEP)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = tf.nn.static_rnn(rnn_cell, sequence, dtype=tf.float32)

    # there are TIME_STEP outputs but
    # we only want the last output
    weights =  tf.Variable(tf.random_normal([n_hidden, 2]))
    biases = tf.Variable(tf.random_normal([2]))
    pred = tf.matmul(outputs[-1], weights) + biases

    # number of units in RNN cell
    n_hidden = 20

    #RNN LOSS FUNCTION
    #loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y)
    loss = tf.square(tf.losses.absolute_difference(y,pred))
    loss_summary = tf.summary.scalar('loss',loss)

    #RNN OPTIMIZER
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(loss)

    #seq_length = tf.placeholder(tf.int32)
    saver = tf.train.Saver()
    merged = tf.summary.merge([loss_summary])

'''
