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
def get_tiles(img,width=PATCH_WIDTH,height=PATCH_HEIGHT):

    tiles = np.zeros((int(img.shape[0] / height) * int(img.shape[1] / width),height, width,img.shape[2]))
    count = 0

    for i in range(0,img.shape[0],height):

        for j in range(0,img.shape[1],width):
            tile = img[i:i+height,j:j+width]

            if tile.shape[0] == height and tile.shape[1] == width:
                tiles[count] = tile
                count += 1

    return tiles

def undo_tiles(tiles,w,h):
    w = w - (w % PATCH_WIDTH)
    h = h - (h % PATCH_HEIGHT)
    canvas = np.zeros((h,w),dtype=np.float32)
    count = 0
    for i in range(0,h,PATCH_HEIGHT):
        for j in range(0,w,PATCH_WIDTH):
            canvas[i:i+PATCH_HEIGHT,j:j+PATCH_WIDTH] = tiles[count]
            count += 1

    return canvas

#########################################################################################################
#MODEL DEFINITION
##########################################################################################################
with tf.variable_scope('cam_model_small'):
    with tf.variable_scope('param_predictor'):

        #input are optical flow vectors as patches
        x = tf.placeholder(tf.float32,[None,PATCH_HEIGHT,PATCH_WIDTH,DEPTH])
        #x_reshaped = tf.reshape(x,[-1,PATCH_WIDTH * PATCH_HEIGHT,DEPTH])
        #vecs = tf.split(x_reshaped,2,axis=2)
        GVx,GVy = tf.split(x,2,axis=-1)

        w1 = tf.Variable(tf.random_normal([13,13,2,32],dtype=tf.float32),dtype=tf.float32)
        w2 = tf.Variable(tf.random_normal([7,7,32,10],dtype=tf.float32),dtype=tf.float32)
        w3 = tf.Variable(tf.random_normal([3,3,10,10],dtype=tf.float32),dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal([32],dtype=tf.float32),dtype=tf.float32)
        b2 = tf.Variable(tf.random_normal([10],dtype=tf.float32),dtype=tf.float32)
        b3 = tf.Variable(tf.random_normal([10],dtype=tf.float32),dtype=tf.float32)

        conv1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
        conv1_out = tf.nn.sigmoid(tf.nn.bias_add(conv1,b1))
        conv2 = tf.nn.conv2d(conv1_out,w2,strides=[1,1,1,1],padding='SAME')
        conv2_out = tf.nn.sigmoid(tf.nn.bias_add(conv2,b2))
        conv3 = tf.nn.conv2d(conv2_out,w3,strides=[1,1,1,1],padding='SAME')
        conv3_out = tf.nn.bias_add(conv3,b3)

        #clip the values to a range
        Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f = tf.split(conv3_out,10,axis=-1)
        Z = tf.clip_by_value(Z,1,1000)
        f = tf.clip_by_value(f,1,1000)
        Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f = [tf.clip_by_value(val,-1000,1000) for val in [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f]]

        #compute the loss using the parameters found
        trans_x = tf.divide(tf.subtract(tf.multiply(Tz,X),tf.multiply(Tx,f)),Z)
        trans_y = tf.divide(tf.subtract(tf.multiply(Tz,Y),tf.multiply(Ty,f)),Z)
        rotation_x = tf.multiply(Wy,f) + tf.multiply(Wz,Y) + tf.divide(tf.multiply(tf.multiply(Wx,X),Y),f) - tf.divide(tf.multiply(tf.multiply(Wy,X),X),f)
        rotation_y = tf.multiply(Wx,f) - tf.multiply(Wz,X) - tf.divide(tf.multiply(tf.multiply(Wy,X),Y),f) + tf.divide(tf.multiply(tf.multiply(Wx,Y),Y),f)
        Vx = trans_x + rotation_x
        Vy = trans_y - rotation_y

        #our loss function is a simple DISTANCE FUNCTION FROM TARGET OUTPUT
        #IF YOU NOTICE, OUR OUTPUT IS TRYING TO BE THE SAME AS OUR INPUT...
        #THIS ONLY WORKS BECAUSE OF THE COMPLEX MODEL WE FORCE THE NEURAL NETWORK TO LEARN THE PARAMETERS FOR
        #WE ARE HOPING THAT THE NEURAL NETWORK LEARNS TO DETERMINE THE PARAMETERS FOR THE FUNCTION WHICH CAN OUTPUT THE CORRECT CAMERA MOTION GIVEN OPTICAL FLOW VECTORS OF STILL OBJECTS
        lossx = tf.abs(tf.subtract(Vx,GVx))
        lossy = tf.abs(tf.subtract(Vy,GVy))
        cost = tf.reduce_mean(tf.add(lossx,lossy))
        opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

        A = [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f,lossx,lossy]
        for i,val in enumerate(A):
            A[i] = tf.cast(tf.divide(tf.subtract(val,tf.reduce_min(val)),tf.reduce_max(val)),tf.float32)

        loss_summary = tf.summary.scalar('loss',cost)
        summs = [loss_summary]
        for name,tens in zip(['Tx','Ty','Tz','Wx','Wy','Wz','X','Y','Z','f','lossx','lossy'],A):
            summary = tf.summary.image(name,tens)
            summs.append(summary)

saver = tf.train.Saver()
merged = tf.summary.merge([summ for summ in summs])

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

    input_flow = tools.readflofile(flow_path)
    h,w,d = input_flow.shape

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if ckpt_path != '':
        saver.restore(sess,ckpt_path)

    myvars = [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f,Vx,Vy,lossx,lossy,cost]
    myvar_names = ["Tx","Ty","Tz","Wx","Wy","Wz","X","Y","Z","f","Vx","Vy",'lossx','lossy',"cost"]

    output = sess.run(myvars, feed_dict={x: input_flow.reshape((1,h,w,d))})

    #VISUALIZE TRAINED PARAMETERS
    for val,val_name in zip(output[:-1],myvar_names[:-1]):
        val -= np.amin(val)
        val = val / np.amax(val) * 255
        cv2.imshow(val_name,val.reshape((h,w)).astype(np.uint8))

    cv2.waitKey(0)
    print("my loss is %.4f" % output[-1])

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

        [shutil.rmtree(os.path.join("log/cam_model_small",f)) for f in os.listdir('log/cam_model_small')]

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
