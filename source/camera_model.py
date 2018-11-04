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
from matplotlib import pyplot as plt



#################################################################################################################################################
#################################################################################################################################################
#DEFINE OUR LOGGER

tf.logging.set_verbosity(tf.logging.INFO)

#PREDIFINED CONSTRAINTS
EPOCH=3
BATCH_STEP=1000
LEARNING_RATE = 0.0001
HEIGHT = 436
WIDTH = 1024
DEPTH = 3
PATCH_HEIGHT = 3
PATCH_WIDTH = 3
T_DIR1 = './data/mountain_1'
T_DIR2 = './data/sleeping_1'
T_DIR3 = './data/sleeping_2'
V_DIR = './data/sleeping_1'
all_dir = "/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow"
#################################################################################################################################################
#################################################################################################################################################
def gather_cols(params, indices, name=None):
    """Gather columns of a 2D tensor.

    Args:
        params: A 2D tensor.
        indices: A 1D tensor. Must be one of the following types: ``int32``, ``int64``.
        name: A name for the operation (optional).

    Returns:
        A 2D Tensor. Has the same type as ``params``.
    """
    with tf.op_scope([params, indices], name, "gather_cols") as scope:
        # Check input
        params = tf.convert_to_tensor(params, name="params")
        indices = tf.convert_to_tensor(indices, name="indices")
        try:
            params.get_shape().assert_has_rank(2)
        except ValueError:
            raise ValueError('\'params\' must be 2D.')
        try:
            indices.get_shape().assert_has_rank(1)
        except ValueError:
            raise ValueError('\'params\' must be 1D.')

        # Define op
        p_shape = tf.shape(params)
        p_flat = tf.reshape(params, [-1])
        i_flat = tf.reshape(tf.reshape(tf.range(0, p_shape[0]) * p_shape[1],
                                       [-1, 1]) + indices, [-1])
        return tf.reshape(tf.gather(p_flat, i_flat),
                          [p_shape[0], -1])

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
#OUR GLOBAL STEP
global_step = tf.Variable(0, name='global_step', trainable=False)
inc = tf.assign_add(global_step,1)

#HELPER FUNCTION
def camera_model_single(input_):
    #input are optical flow vectors of still objects
    #intialize our weights and biases
    w1 = tf.Variable(tf.random_normal([2,200],dtype=tf.float32),dtype=tf.float32)
    w2 = tf.Variable(tf.random_normal([200,200],dtype=tf.float32),dtype=tf.float32)
    w3 = tf.Variable(tf.random_normal([200,10],dtype=tf.float32),dtype=tf.float32)
    b1 = tf.Variable(tf.random_normal([200],dtype=tf.float32),dtype=tf.float32)
    b2 = tf.Variable(tf.random_normal([200],dtype=tf.float32),dtype=tf.float32)
    b3 = tf.Variable(tf.random_normal([10],dtype=tf.float32),dtype=tf.float32)

    #our model internals
    l1 = tf.add(tf.matmul(input_,w1),b1)
    l1_out = tf.nn.sigmoid(l1)
    l2 = tf.add(tf.matmul(l1_out,w2),b2)
    l2_out = tf.nn.sigmoid(l2)
    l3 = tf.add(tf.matmul(l2_out,w3),b3)

    Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f = tf.split(l3,10,axis=1)
    Z = tf.clip_by_value(Z,1,100)
    f = tf.clip_by_value(f,1,100)
    a = [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f]
    for i,tensor in enumerate(a):
        a[i] = tf.clip_by_value(tensor,-100,100)

    trans_x = tf.divide(tf.subtract(tf.multiply(Tz,X),tf.multiply(Tx,f)),Z)
    trans_y = tf.divide(tf.subtract(tf.multiply(Tz,Y),tf.multiply(Ty,f)),Z)
    rotation_x = tf.multiply(Wy,f) + tf.multiply(Wz,Y) + tf.divide(tf.multiply(tf.multiply(Wx,X),Y),f) - tf.divide(tf.multiply(tf.multiply(Wy,X),X),f)
    rotation_y = tf.multiply(Wx,f) - tf.multiply(Wz,X) - tf.divide(tf.multiply(tf.multiply(Wy,X),Y),f) + tf.divide(tf.multiply(tf.multiply(Wx,Y),Y),f)
    Vx = trans_x + rotation_x
    Vy = trans_y - rotation_y
    GVx = gather_cols(input_,np.array([0],dtype=np.int32))
    GVy = gather_cols(input_,np.array([1],dtype=np.int32))

    #our loss function is a simple DISTANCE FUNCTION FROM TARGET OUTPUT
    #IF YOU NOTICE, OUR OUTPUT IS TRYING TO BE THE SAME AS OUR INPUT...
    #THIS ONLY WORKS BECAUSE OF THE COMPLEX MODEL WE FORCE THE NEURAL NETWORK TO LEARN THE PARAMETERS FOR
    #WE ARE HOPING THAT THE NEURAL NETWORK LEARNS TO DETERMINE THE PARAMETERS FOR THE FUNCTION WHICH CAN OUTPUT THE CORRECT CAMERA MOTION GIVEN OPTICAL FLOW VECTORS OF STILL OBJECTS
    lossx = tf.abs(tf.subtract(Vx,GVx))
    lossy = tf.abs(tf.subtract(Vy,GVy))
    cost = tf.reduce_mean(tf.add(lossx,lossy))
    opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    loss_sum = tf.summary.scalar('loss',cost)
    f_sum = tf.summary.histogram('f',f)
    z_sum = tf.summary.histogram('z',Z)

    return {"params": a, "loss": cost, "opt": opt, "loss_sum": loss_sum, "f_sum": f_sum, "z_sum": z_sum}

#our model
with tf.variable_scope('cam_model'):
    with tf.variable_scope('param_predictor'):

        #input are optical flow vectors of still objects
        x = tf.placeholder(tf.float32,[None,PATCH_HEIGHT,PATCH_WIDTH,2])
        x_reshaped = tf.reshape(x,(-1,PATCH_HEIGHT * PATCH_WIDTH,2))
        vecs = tf.split(x_reshaped,PATCH_HEIGHT * PATCH_WIDTH,axis=1)

        #otuputs
        outputs = []
        values = []
        for v in vecs:
            v = tf.reshape(v,[-1,2])
            params = camera_model_single(v)
            values.append(params)
            outputs += params["params"]

        #intialize our weights and biases
        w1 = tf.Variable(tf.random_normal([PATCH_WIDTH * PATCH_HEIGHT * 10,300],dtype=tf.float32),dtype=tf.float32)
        w2 = tf.Variable(tf.random_normal([300,300],dtype=tf.float32),dtype=tf.float32)
        w3 = tf.Variable(tf.random_normal([300,10],dtype=tf.float32),dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal([300],dtype=tf.float32),dtype=tf.float32)
        b2 = tf.Variable(tf.random_normal([300],dtype=tf.float32),dtype=tf.float32)
        b3 = tf.Variable(tf.random_normal([10],dtype=tf.float32),dtype=tf.float32)

        #concatenate our list of parameters from our network
        outputs = tf.concat(outputs,axis=1)

        #our model internals
        l1 = tf.add(tf.matmul(outputs,w1),b1)
        l1_out = tf.nn.sigmoid(l1)
        l2 = tf.add(tf.matmul(l1_out,w2),b2)
        l2_out = tf.nn.sigmoid(l2)
        l3 = tf.add(tf.matmul(l2_out,w3),b3)

        #our physics equation to calculate optical flow
        #outputs are the paremeters for our optical flow equation
        #tx,ty,tz,wx,wy,wz,x,y,z
        Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f = tf.split(l3,10,axis=1)
        Z = tf.clip_by_value(Z,1,100)
        f = tf.clip_by_value(f,1,100)
        a = [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f]
        for i,tensor in enumerate(a):
            a[i] = tf.clip_by_value(tensor,-100,100)

        trans_x = tf.divide(tf.subtract(tf.multiply(Tz,X),tf.multiply(Tx,f)),Z)
        trans_y = tf.divide(tf.subtract(tf.multiply(Tz,Y),tf.multiply(Ty,f)),Z)
        rotation_x = tf.multiply(Wy,f) + tf.multiply(Wz,Y) + tf.divide(tf.multiply(tf.multiply(Wx,X),Y),f) - tf.divide(tf.multiply(tf.multiply(Wy,X),X),f)
        rotation_y = tf.multiply(Wx,f) - tf.multiply(Wz,X) - tf.divide(tf.multiply(tf.multiply(Wy,X),Y),f) + tf.divide(tf.multiply(tf.multiply(Wx,Y),Y),f)
        Vx = trans_x - rotation_x
        Vy = trans_y + rotation_y

        #our loss function is a simple DISTANCE FUNCTION FROM TARGET OUTPUT
        #IF YOU NOTICE, OUR OUTPUT IS TRYING TO BE THE SAME AS OUR INPUT...
        #THIS ONLY WORKS BECAUSE OF THE COMPLEX MODEL WE FORCE THE NEURAL NETWORK TO LEARN THE PARAMETERS FOR
        #WE ARE HOPING THAT THE NEURAL NETWORK LEARNS TO DETERMINE THE PARAMETERS FOR THE FUNCTION WHICH CAN OUTPUT THE CORRECT CAMERA MOTION GIVEN OPTICAL FLOW VECTORS OF STILL OBJECTS
        GVx = tf.gather(x_reshaped,[0],axis=-1)
        GVy = tf.gather(x_reshaped,[1],axis=-1)
        GVx = tf.reshape(GVx,(-1,PATCH_WIDTH * PATCH_HEIGHT))
        GVy = tf.reshape(GVy,(-1,PATCH_WIDTH * PATCH_HEIGHT))

        lossx = tf.abs(tf.subtract(Vx,GVx))
        lossy = tf.abs(tf.subtract(Vy,GVy))
        lossx_out = tf.reduce_mean(lossx,axis=1)
        lossy_out = tf.reduce_mean(lossy,axis=1)
        cost = tf.reduce_mean(tf.add(lossx,lossy))

        optlist = [w1,w2,w3,b1,b2,b3]
        opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost,var_list=optlist)

        loss_summary = tf.summary.scalar('loss',cost)
        Tx_summary = tf.summary.histogram('Tx',Tx)
        Ty_summary = tf.summary.histogram('Ty',Ty)
        Tz_summary = tf.summary.histogram('Tz',Tz)
        Wx_summary = tf.summary.histogram('Wx',Wx)
        Wy_summary = tf.summary.histogram('Wy',Wy)
        Wz_summary = tf.summary.histogram('Wz',Wz)
        X_summary = tf.summary.histogram('X',X)
        Y_summary = tf.summary.histogram('Y',Y)
        Z_summary = tf.summary.histogram('Z',Z)
        f_summary = tf.summary.histogram('f',f)

        summaries = [loss_summary,Tx_summary,Ty_summary,Tz_summary,Wx_summary,Wy_summary,Wz_summary,X_summary,Y_summary,Z_summary,f_summary]
        summaries1 = []

for val in values:
    summaries1.append(val["loss_sum"])
    summaries1.append(val["z_sum"])
    summaries1.append(val["f_sum"])

saver = tf.train.Saver()
merged1 = tf.summary.merge(summaries1)
merged = tf.summary.merge(summaries)

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

    print(ckpt_path)

    input_flow = tools.readflofile(flow_path)
    tiles = get_tiles(input_flow)

    h,w,d = input_flow.shape
    x_in = input_flow.reshape((h*w,d))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    if ckpt_path != '':
        saver.restore(sess,ckpt_path)

    myvars = [Tx,Ty,Tz,Wx,Wy,Wz,X,Y,Z,f,Vx,Vy,lossx_out,lossy_out,cost]
    myvar_names = ["Tx","Ty","Tz","Wx","Wy","Wz","X","Y","Z","f","Vx","Vy","lossx","lossy","cost"]

    output = sess.run(myvars, feed_dict={x: tiles})

    #VISUALIZE TRAINED PARAMETERS
    for val,val_name in zip(output[:-1],myvar_names[:-1]):
        img = undo_tiles(val,w,h)
        img -= np.amin(img)
        img = img / np.amax(img) * 255
        cv2.imshow(val_name,img.astype(np.uint8))

    cv2.waitKey(0)

    print("my loss is %.4f" % output[-1])

##########################################################################################################
# define our training module
#GET OUR TESTING BATCH
def getTBatch(file_list):
    tmp = []
    count = 0
    for f in file_list:
        flowfield = tools.readflofile(f)
        tiles = get_tiles(flowfield)
        tmp.append(tiles)
        count += tiles.shape[0]

    train_x = np.zeros((count,PATCH_HEIGHT,PATCH_WIDTH,2))
    cur_count = 0
    for i,field in enumerate(tmp):
        length = field.shape[0]
        train_x[cur_count: cur_count + length] = field
        cur_count += length

    return train_x

#training / testing cross validation
def train(ckpt_path='',mode='still',name='cam_model'):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #restore model to ckpt if we have one
        if ckpt_path != '':
            saver.restore(sess,ckpt_path)

        #training/testing logger
        if not os.path.isdir("./log/cam_model"):
            os.mkdir("./log/cam_model")
        tools.clearFile("./log/cam_model")
        training_writer = tf.summary.FileWriter('./log/cam_model/training',sess.graph)
        testing_writer = tf.summary.FileWriter('./log/cam_model/testing',sess.graph)

        modelout = os.path.join('model',name) + '.ckpt'

        #my training data as file names
        training_list = []
        if mode == 'still':
            training_list.append([os.path.join(T_DIR1,fname) for fname in os.listdir(T_DIR1)])
            training_list.append([os.path.join(T_DIR2,fname) for fname in os.listdir(T_DIR2)])
            training_list.append([os.path.join(T_DIR3,fname) for fname in os.listdir(T_DIR3)])
        elif mode == 'all':
            for flow_dir in os.listdir(all_dir):
                training_list.append([os.path.join(all_dir,flow_dir,flow_path) for flow_path in os.listdir(os.path.join(all_dir,flow_dir))])

        #our training steps
        counter = 0
        for i in range(EPOCH):
            for j in range(len(training_list)):
                #training batch getter
                train_x = getTBatch(training_list[j])

                #go through our steps
                for s in range(0,len(train_x),BATCH_STEP):

                    #merge our summaries
                    losses = []
                    for param in values:
                        summary = sess.run([param["loss"],param["opt"],merged1],feed_dict={x: train_x[s:s+BATCH_STEP]})
                        losses.append(summary[0])

                    saver.save(sess,modelout)
                    training_writer.add_summary(summary[-1],global_step=counter)
                    counter += 1

                print("epoch: %i batch %i cost %.4f" % (i,j,summary[0]))

        counter = 0
        for i in range(EPOCH):
            for j in range(len(training_list)):
                #training batch getter
                train_x = getTBatch(training_list[j])

                #go through our steps
                for s in range(0,len(train_x),BATCH_STEP):

                    summary = sess.run([cost,opt,merged],feed_dict={x: train_x[s:s+BATCH_STEP]})
                    saver.save(sess,modelout)
                    training_writer.add_summary(summary[-1],global_step=counter)
                    counter += 1

                print("epoch: %i batch %i cost %.4f" % (i,j,summary[0]))

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
        name = 'cam_model'
        if '-model' in sys.argv:
            model = sys.argv[sys.argv.index('-model') + 1]
        if '-input' in sys.argv:
            mode = sys.argv[sys.argv.index('-input') +1]
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




