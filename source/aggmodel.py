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


##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

def normalize(x):
    return tf.divide(tf.subtract(x,tf.reduce_min(x)),tf.subtract(tf.reduce_max(x),tf.reduce_min(x)))


def bird_model(input_):
    n_hidden = 100

    cell = tf.nn.rnn_cell.LSTMCell(100)

    outputs, states = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32)

    w = tf.Variable(tf.random_normal([n_hidden,2]))
    b = tf.Variable(tf.random_normal([2]))

    return tf.matmul(outputs[:,-1,:],w) + b



##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

##########################################################################################################
#MODEL DEFINITION
##########################################################################################################
#OUR GLOBAL STEP



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
        train_x,train_y = getTBatch(training_list)

        final = tools.readflofile(training_list[0])
        training = tools.readflofile(training_list[1])

        #tools.showVec(vecs=[train_x[:,:,i,:] for i in range(4)])
        h,w,t,d = train_x.shape

        #predict future trajectories
        out, lossval, lossmap = sess.run([outputs,loss,diff],feed_dict={x: train_x.reshape((h * w,t,d)), y: train_y.reshape((h*w,d))})

        new_pos = out.reshape((h,w,d))
        new_pos[:,:,0] = new_pos[:,:,0] * w
        new_pos[:,:,1] = new_pos[:,:,1] * h

        print("lossmap bounds: ",np.amin(lossmap),np.amax(lossmap))
        #warped = tools.customWarp(train_x[:,:,-1,:],vec)
        tools.map_new(training,final,new_pos)

        vec1 = tools.readflofile(training_list[1])
        tools.showVec(vecs=[vec,train_x[:,:,-1,:],train_y,warped])

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

    flow_map_x, pos_map_x = stitch_flo_files(files[1:])

    forward_data = files[:2]
    forward_data.reverse()
    flow_map_y, pos_map_y = stitch_flo_files(forward_data,mode='forward')

    return pos_map_x, pos_map_y[:,:,0,:]

def stitch_flo_files(files,mode='back'):
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

    return tools.stitchFlow(flows,mode=mode)

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

                print('epoch: %i loss: %.4f ' % (i+1,lossvalue))

#training / testing cross validation
def train2(ckpt_path='',mode='real'):
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

# MAIN FUNCTION
if __name__ == '__main__':
    x = tf.placeholder(tf.float32,shape=[None,TIME_STEP,2])
    x = normalize(x)
    y = tf.placeholder(tf.float32,shape=[None,2])
    y = normalize(y)

    outputs= bird_model(x)

    #LOSS FOR THE BIRD MODEL
    diff = tf.reduce_sum(tf.square(tf.subtract(y,outputs)),axis=-1)
    loss = tf.reduce_mean(diff)
    loss_summary = tf.summary.scalar('loss',loss)

    #RNN OPTIMIZER
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LR).minimize(loss)

    #seq_length = tf.placeholder(tf.int32)
    saver = tf.train.Saver()
    merged = tf.summary.merge([loss_summary])
    if '-train' in sys.argv:
        model = ''
        ex_data = 'real'
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        if '-example' in sys.argv:
            ex_data = sys.argv[sys.argv.index('-example') + 1]

        train(ckpt_path=model,mode=ex_data)
    if '-test' in sys.argv:
        model = ''
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        else:
            print('YOU NEED A MODEL TO TEST ON...')

        test(sys.argv[sys.argv.index('-test') + 1],ckpt_path=model)

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
