import cv2
import numpy as np
import os
import sys
import random
import shutil		#for removing file directories we are logging into
import time
from matplotlib import pyplot as plt
from scipy import ndimage

###################################################################
####################################################################################################################################
#REMOVES THE CONTENTS OF A FILE COMPLETELY. USE AT YOUR OWN RISK
def clearFile(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

####################################################################################################################################
#READS A .FLO FILE AND RETURNS THE 2D DATASET. -1 IF THE FILE COULD NOT BE READ
#INPUT: STR
#OUTPUT: NUMPY ARRAY OF SHAPE (h,w,2)
def readflofile(flofile):
    if not os.path.exists(flofile):
        print('file does not exist')
        return -1
    with open(flofile,'rb') as f:
        magic = np.fromfile(f,np.float32,count=1)
        if 202021.25 == magic:

            w = np.fromfile(f,np.int32,count=1)[0]
            h = np.fromfile(f,np.int32,count=1)[0]

            data = np.fromfile(f,np.float32,count=2*w*h)
            data2D = np.resize(data,(h,w,2))
            return data2D
        else:
            return -1

#function for tiling an image into patches of size w * h
def get_tiles(img,width=5,height=5):
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

####################################################################################################################################
#GENERATE VIDEO SEQUENCE FROM SEQUENCE OF IMAGES AND THE FLOW MAP
def genFlowSeq(img,flowseq):
    h,w,d = img.shape
    canvas = img.copy()
    canvas.fill(0)
    seqlength = flowseq.shape[2]
    initial_pos = np.dstack(np.meshgrid(np.arange(w),np.arange(h)))
    #video = cv2.VideoWriter("flowseq.avi",cv2.VideoWriter_fourcc(),1,(w,h))

    #tmp = initial_pos >= 0
    #boundry = tmp[:,:,0]

    #canvas[:,:,0].reshape((h*w))[idx] = img[:,:,0].reshape((h*w)).copy()
    #canvas[:,:,1].reshape((h*w))[idx] = img[:,:,1].reshape((h*w)).copy()
    #canvas[:,:,2].reshape((h*w))[idx] = img[:,:,2].reshape((h*w)).copy()

    #loop through each flow and map the current pixels backwards in time
    for i in reversed(range(seqlength)):
        flow = flowseq[:,:,i,:]

        cur_map = np.rint(initial_pos + flow)

        #FIND THE NEW BOUNDRIES I.E SPOTS WHERE THE CURMAP FALL OFF THE IMAGE
        leftbound = np.logical_and(cur_map[:,:,0] >= 0,cur_map[:,:,1] >= 0)
        rightbound = np.logical_and(cur_map[:,:,0] < w-1,cur_map[:,:,1] < h-1)
        boundry = np.logical_and(leftbound,rightbound)

        idx = (cur_map[boundry][:,1] * (w)) + cur_map[boundry][:,0]
        idx = idx.astype(np.uint32)

        canvas[:,:,0].reshape((h*w))[idx] = img[boundry][:,0].copy()
        canvas[:,:,1].reshape((h*w))[idx] = img[boundry][:,1].copy()
        canvas[:,:,2].reshape((h*w))[idx] = img[boundry][:,2].copy()

        cv2.imshow('Sequence',canvas.astype(np.uint8))
        cv2.waitKey(0)
        canvas.fill(0)

        #video.write(canvas.astype(np.uint8))
        initial_pos = cur_map.copy()

    #cv2.destroyAllWindows()
    #video.release()

####################################################################################################################################
def saveflo(flow,filename):
    with open(filename,'wb') as fout:
        np.array([ 80, 73, 69, 72 ], np.uint8).tofile(fout)
        np.array([flow.size(2), flow.size(1)], np.int32).tofile(fout)
        np.array(flow[[1,0]], np.float32).transpose(1,2,0).tofile(fout)
####################################################################################################################################

#TAKES 2 VECTORS AND RETURNS THE SCALAR VECTOR DIFFERENCE. AKA END POINT ERROR (EPE)
def calcEPE(vec1,vec2):
    if vec1.shape[-1] == vec2.shape[-1] and vec1.shape[-1] == 2:
        vec = np.power(vec1 - vec2,2)
        vecsum = vec[:,:,0] + vec[:,:,1]

        return np.sqrt(vecsum)
####################################################################################################################################

#VISUALIZE VEC OF SHAPE (H,W,2)
# Use Hue, Saturation, Value colour model
#https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
def showVec(vecs,waitkey=0,name='colored flow'):

    for i,vec in enumerate(vecs):
        hsv = np.zeros((vec.shape[0],vec.shape[1],3), dtype=np.uint8)
        hsv[:,:, 1] = 255

        mag, ang = cv2.cartToPolar(vec[:,:, 0], vec[:,:, 1])
        print(mag,ang)
        hsv[:,:, 0] = ang * 180 / np.pi / 2
        hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        #cv2.namedWindow(name + str(i),cv2.WINDOW_NORMAL)
        cv2.imshow(name + str(i), bgr)

    cv2.waitKey(waitkey)
####################################################################################################################################
#WARP THE IMAGE t to t+s USING OPTICAL FLOW t
def warpImgwFlow(img, flow,direction='forward'):
    h,w,d = img.shape

    R2 = np.dstack(np.meshgrid(np.arange(w),np.arange(h)))
    if direction == 'forward':
        vecmap = R2 - flow
    else:
        vecmap = R2 + flow

    grid = [vecmap[:,:,1],vecmap[:,:,0]]

    warp_im = np.empty((h,w,d),dtype=np.float32)

    for i in range(d):
        warp_im[:,:,i]= ndimage.map_coordinates(img[:,:,i], grid, order=5,mode='constant',prefilter=False)

    return warp_im

def customWarp(img,flow,direction='forward'):
    h,w,d = img.shape

    canvas = img.copy()
    canvas.fill(0)
    initial_pos = np.dstack(np.meshgrid(np.arange(w),np.arange(h)))

    #loop through each flow and map the current pixels forward in time
    cur_map = np.rint(initial_pos + flow)

    #FIND THE NEW BOUNDRIES I.E SPOTS WHERE THE CURMAP FALL OFF THE IMAGE
    leftbound = np.logical_and(cur_map[:,:,0] >= 0,cur_map[:,:,1] >= 0)
    rightbound = np.logical_and(cur_map[:,:,0] < w,cur_map[:,:,1] < h)
    boundry = np.logical_and(leftbound,rightbound)

    idx = (cur_map[boundry][:,1] * (w)) + cur_map[boundry][:,0]
    idx = idx.astype(np.uint32)

    canvas[:,:,0].reshape((h*w))[idx] = img[boundry][:,0].copy()
    canvas[:,:,1].reshape((h*w))[idx] = img[boundry][:,1].copy()

    return canvas

def map_new(orig,final,new_map):

    h,w,d = orig.shape
    R2 = np.dstack(np.meshgrid(np.arange(w),np.arange(h)))
    flow = new_map - R2

    tmp = orig[200:300,400:500,:]
    tmp2 = final[200:300,400:500,:]
    tmp3 = new_map[200:300,400:500,:]
    tmp4 = R2[200:300,400:500,:]
    f_tmp = tmp3 - tmp4

    tmp5 = tmp4.copy()

    tmp5[:,:,0] = tmp4[:,:,0] - 400
    tmp5[:,:,1] = tmp4[:,:,1] - 200

    new_img = warpImgwFlow(tmp5,f_tmp)
    new_img2 = warpImgwFlow(orig,flow)
    showVec([orig,final,tmp,new_img,new_img2])

    quit()

####################################################################################################################################
#FIND PIXEL TO PIXEL TEMPORAL CORRESPONDANCE ACROSS A VIDEO FRAME SEQUENCE
#INPUT:
#   OPTICAL FLOW MAP FOR EACH SEQUENCE
#   DEPTH MAP FOR EACH SEQUENCE
#   IMG MAP FOR EACH SEQUENCE (OPTIONAL)
#OUTPUT:
#   PIXEL CORRESPONDANCE OF FINAL FRAME IN THE VIDEO SEQUENCE AS NUMPY ARRAY
#   OF SHAPE (H,W,N,5)
#   WHERE VALUES ARE [Fx,Fy,pos_i,pos_j,depth]
#
def stitchFlow(f_maps,d_maps,img_map):
    n,h,w,d = f_maps.shape

    #INITIALIZE OUTPUT DATA
    out = np.zeros((h,w,n,5))

    #INITIALIZE SOME DATA STRUCTURES FOR THE MAPPING
    cur_map = np.dstack(np.meshgrid(np.arange(w),np.arange(h)) + [d_maps[0,:,:]]).astype(np.float32)
    active_threads = np.ones((h,w)) == 1        #ALL THREADS START OFF ACTIVE
    grid = np.dstack(np.meshgrid(np.arange(w),np.arange(h))).astype(np.uint32)  #DEFAULT LABEL NAMES
    positions = grid.reshape((h*w,2))[:,1] * w + grid.reshape((h*w,2))[:,0]
    indices = set(positions)
    #STITCH OPTICAL FLOW FORWARD
    #ALWAYS MAP LABEL LOCATIONS TO PIXEL LOCATIONS. LABEL LOCATIONS ARE THE SAME AS PIXEL LOCATIONS STARTING OFF.
    for i,maps in enumerate(zip(f_maps,d_maps,img_map)):
        #INITIALIZE THE DATA FOR CURRENT FRAME
        flow,depth,img = maps

        #SAVE CURRENT STATE INFORMATION TO THE ACTIVE THREADS
        #update valid threads and initialize loop requirements
        #we also use the depth map of each pixel to determine mapping procedure of each thread
        elevation = np.unique(cur_map[:,:,2].astype(np.int32))  #pixel depth information
        gate_way = []   #container for mapped/unmapped pixels, so we only have one label per pixel
        for e in elevation:
            #GET THE THREADS THAT STAYED WITHIN BOUNDS IN THE UPDATED MAP.
            #FIX INTERSECTIONS BY LOOKING AT DEPTH
            leftbound = np.logical_and(np.logical_and(cur_map[:,:,0] >= 0,cur_map[:,:,1] >= 0),cur_map[:,:,2].astype(np.uint32) == e)
            rightbound = np.logical_and(np.logical_and(cur_map[:,:,1] < h-1,cur_map[:,:,0] < w-1),cur_map[:,:,2].astype(np.uint32) == e)
            boundry = np.logical_and(leftbound,rightbound)  #threads which are still within bounds

            #GATHER INDICES USING MAPPED PIXEL POSITIONS + FLOW
            idx = cur_map[boundry][:,1].astype(np.uint32) * w + cur_map[boundry][:,0].astype(np.uint32) #img positions
            idx_label = grid[boundry][:,1].astype(np.uint32) * w + grid[boundry][:,0].astype(np.uint32)     #label positions

            #TAKE THE LOGICAL INVERSE OF IS IN
            idx_mask = np.logical_not(np.isin(idx, gate_way))       #IF THE IDX HAS BEEN DISCOVERED BEFORE, IT WILL APPEAR IN GATEWAY

            #UPDATE ONE THREAD PER INTERSECTION
            out[:,:,i,:2].reshape((h*w,2))[idx_label[idx_mask]] = flow.reshape((h*w,2))[idx[idx_mask]]
            out[:,:,i,2:4].reshape((h*w,2))[idx_label[idx_mask]] = cur_map.reshape((h*w,3))[idx_label[idx_mask]][:,:2]
            out[:,:,i,4].reshape((h*w))[idx_label[idx_mask]] = depth.reshape((h*w))[idx[idx_mask]]

            #UPDATE THE MEMORY GATE
            gate_way += list(idx)

        #CHECK HOLES AND NON HOLES
        #FIX THE HOLES THAT WERE CREATED WHEN PUSHING THE MAP WITH OPTICAL FLOW
        holes = np.all(out[:,:,i,:] == [0,0,0,0,0],axis=-1)
        non_holes = np.logical_not(holes)
        holes_flat = holes.reshape((h*w))

        idx = indices - set(gate_way)
        min_val = int(min(len(idx),np.count_nonzero(holes)))      #TAKE THE MIN LENGTH BETWEEN THE NUMBER OF HOLES AND THE NUMBER OF AVAILABLE LABELS
        out[holes] = 0                                #RESET NON-ACTIVE THREADS
        out[:,:,i,:2].reshape((h*w,2))[positions[holes_flat][:min_val]] = flow.reshape((h*w,2))[list(idx)][:min_val]
        out[:,:,i,2:4].reshape((h*w,2))[positions[holes_flat][:min_val]] = grid.reshape((h*w,2))[list(idx)][:min_val]
        out[:,:,i,4].reshape((h*w))[positions[holes_flat][:min_val]] = depth.reshape((h*w))[list(idx)][:min_val]

        #UPDATE THE ACTIVE THREADS ACCORDING TO UPDATED THREADS
        holes = np.all(out[:,:,i,:] == [0,0,0,0,0],axis=-1)
        non_holes = np.logical_not(holes)
        active_threads = np.logical_and(non_holes,active_threads)

        #UPDATE THE MAP FOR THE NEXT ITERATION. MAKE SURE TO UPDATE THE ACTIVE THREADS ONLY
        idx = np.rint(cur_map.reshape((h*w,3))[:,1] * w + cur_map.reshape((h*w,3))[:,0]).astype(np.int32)
        pos_mask = idx >= 0           #mask for pos values
        bound_mask = idx < h*w    #mask for inbounds values
        bound = np.logical_and(pos_mask,bound_mask) #combined mask
        idx_label = np.rint(grid.reshape((h*w,2))[:,1] * w + grid.reshape((h*w,2))[:,0]).astype(np.int32)   #get labels of current active threads

        #SET THE CURRENT MAP TO NEXT MAP BY ITS CORRESPONDING OPTICAL FLOW
        spot = np.where(idx_label == 100 * w + 400)

        #print(out[100,400],flow.reshape((h*w,2))[idx[spot[0]]])
        cur_map.reshape((h*w,3))[:,:2][idx_label[bound]] = cur_map.reshape((h*w,3))[:,:2][idx_label[bound]] + out[:,:,i,:2].reshape((h*w,2))[idx_label[bound]] #push current map to the next map for next iteration

        #MAKE SURE TO SET THE DEPTH OF THE NEXT MAP TO THE NEXT ITERATION OF THE LOOP. OTHERWISE SET IT TO ITS PREVIOUS VALUE
        if i < n-1: cur_map.reshape((h*w,3))[:,2][idx_label[bound]] = d_maps[i+1].reshape((h*w))[idx[bound]]
        else: cur_map.reshape((h*w,3))[:,2][idx_label[bound]] = out[:,:,i-1,4].reshape((h*w))[idx[bound]]

        '''
        #VISUALIZATION
        for j in range(10):
            for z in range(10):
                x = int(w/10) * j + 40
                y = int(h/10) * z + 20
                c = tuple(out[y,x,i,2:4].astype(np.uint32))
                a = 255 / 10 * j
                b = 255 / 10 * z
                cv2.circle(img,c,5,[a,0,b],thickness=-1)
        c = tuple(out[100,400,i,2:4].astype(np.int32))
        cv2.circle(img,c,10,[255,0,0],thickness=-1)
        cv2.imshow('labeled points',img)
        cv2.waitKey(0)
        '''

    return out

####################################################################################################################################
"""
I/O script to save and load the data coming with the MPI-Sintel low-level
computer vision benchmark.

For more details about the benchmark, please visit www.mpi-sintel.de

CHANGELOG:
v1.0 (2015/02/03): First release

Copyright (c) 2015 Jonas Wulff
Max Planck Institute for Intelligent Systems, Tuebingen, Germany

"""
#READ DEPTH

def read_depth(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename,'rb')
    TAG_FLOAT = 202021.25
    TAG_CHAR = 'PIEH'
    check = np.fromfile(f,dtype=np.float32,count=1)[0]
    assert check == TAG_FLOAT, ' depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT,check)
    width = np.fromfile(f,dtype=np.int32,count=1)[0]
    height = np.fromfile(f,dtype=np.int32,count=1)[0]
    size = width*height
    assert width > 0 and height > 0 and size > 1 and size < 100000000, ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width,height)
    depth = np.fromfile(f,dtype=np.float32,count=-1).reshape((height,width))

    return depth

####################################################################################################################################
#LEARN BACKGROUND?

#PLOT LINE OF BEST FIT GIVEN A SET OF DATAPOINTS
def pltBestFit(pts,mode='quadratic'):
    y = pts[:,0]
    x = pts[:,1]
    t = range(len(pts))
    plt.subplot(221)
    plt.plot(np.unique(t), np.poly1d(np.polyfit(t, y, 2))(np.unique(t)))
    plt.scatter(t,y)
    plt.subplot(222)
    plt.plot(np.unique(t), np.poly1d(np.polyfit(t, x, 2))(np.unique(t)))
    plt.scatter(t,x)
    plt.show()

####################################################################################################################################

if __name__ == '__main__':

    OUT_DIR = 'stuff'
    if 'epe' in sys.argv:
        vec1 = readflofile(sys.argv[2])
        vec2 = readflofile(sys.argv[3])
        epe = calcEPE(vec1,vec2)
        avgEPE = np.mean(epe)
        epe = epe / np.amax(epe) * 255
        print("AVERAGE END POINT ERROR is %.4f" % avgEPE)
        showVec(vec1,waitkey=1,name='Optical flow 1')
        showVec(vec2,waitkey=1,name='Optical flow 2')
        cv2.imshow('end point error visualized',epe.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif 'view' in sys.argv:
        vec =readflofile(sys.argv[2])
        showVec(vec)
    elif 'vdiff' in sys.argv:
        vec1 = readflofile(sys.argv[2])
        vec2 = readflofile(sys.argv[3])
        showVec(vec2-vec1)
    elif 'vimgdiff' in sys.argv:
        img1 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        img2 = cv2.imread(sys.argv[3],cv2.IMREAD_COLOR)
        img3 = img1 - img2
        cv2.imshow('Image Diff', img3.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        fout = 'diff_' + os.path.splitext(os.path.basename(sys.argv[2]))[0] + os.path.splitext(os.path.basename(sys.argv[3]))[0] + '.png'
        cv2.imwrite(os.path.join(OUT_DIR,fout),img3.astype(np.uint8))
    elif 'warp' in sys.argv:
        img1 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        flow = readflofile(sys.argv[3])
        warped_img = warpImgwFlow(img1,flow)

        cv2.imshow('original image',img1)
        cv2.imshow('warped image', warped_img.astype(np.uint8))

        fout = 'warped_' + os.path.basename(sys.argv[2])
        cv2.imwrite(os.path.join(OUT_DIR,fout),warped_img)
        cv2.waitKey(0)

    elif 'backwarp' in sys.argv:
        img1 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        flow = readflofile(sys.argv[3])
        warped_img = warpImgwFlow(img1,flow,direction='backwards')

        cv2.imshow('original image',img1)
        cv2.imshow('warped image', warped_img.astype(np.uint8))

        fout = 'warped_' + os.path.basename(sys.argv[2])
        cv2.imwrite(os.path.join(OUT_DIR,fout),warped_img)
        cv2.waitKey(0)
    elif 'customwarp' in sys.argv:
        img1 = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
        flow = readflofile(sys.argv[3])
        warped_img = customWarpwFlow(img1,flow,direction='backwards')

        cv2.imshow('original image',img1)
        cv2.imshow('warped image', warped_img.astype(np.uint8))

        fout = 'warped_' + os.path.basename(sys.argv[2])
        cv2.imwrite(os.path.join(OUT_DIR,fout),warped_img)
        cv2.waitKey(0)

    elif 'stitch' in sys.argv:   #DEBUGGING OF FUNCTIONS. FEEL FREE TO EDIT AS YOU WISH
        start_time = time.time()
        depth_dir = '/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/MPI-Sintel-depth-training-20150305/training/depth/market_2/'
        flow_dir = '/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/flow/market_2/'
        img_dir = '/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/clean/market_2/'
        first_img = '/media/cvlab/a1716558-819e-4d9f-9a0c-e0fac162c845/cvlab3114/MPI-Sintel-complete/training/clean/market_2/frame_0005.png'
        HEIGHT = 436
        WIDTH = 1024

        #CHECK IF DIRECTORY EXISTS
        if os.path.isdir(depth_dir) and os.path.isdir(flow_dir):
            depth_list = os.listdir(depth_dir)
            flow_list = os.listdir(flow_dir)
            img_list = os.listdir(img_dir)
            depth_list.sort() ; flow_list.sort() ; img_list.sort()
            depth_list = [os.path.join(depth_dir,fname) for fname in depth_list][4:15]
            flow_list = [os.path.join(flow_dir,fname) for fname in flow_list][4:15]
            img_list = [os.path.join(img_dir,fname) for fname in img_list][4:15]

            #READ FIRST 5 FRAMES
            depth_map = np.stack([read_depth(depth_list[i]) for i in range(5)])
            flow_map = np.stack([readflofile(flow_list[i]) for i in range(5)])
            img_map = np.stack([cv2.imread(img_list[i]) for i in range(5)])

            data = stitchFlow(flow_map,depth_map,img_map)

        holes = np.all(data[:,:,0,:] == [0,0,0,0,0],axis=-1)
        print(data[holes])
        print("execution time: %s seconds" % (time.time() - start_time))

    else:
        print("""HERE IS SOME HELP FOR USING THESE TOOLS

        Make sure you are in the same directory of these tools and call it with python2.7 with one of the following options!

        1. [epe]
        2. [view]
        3. [vdiff]
        4. [vimgdiff]
        5. [warp]
        6. [backwarp]
        7. [customwarp] [img_path] [flo_path]
        7. [stitch]

        """)




