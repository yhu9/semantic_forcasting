import cv2
import numpy as np
import os
import sys
import random
import shutil		#for removing file directories we are logging into
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
        hsv[..., 1] = 255

        mag, ang = cv2.cartToPolar(vec[:,:, 0], vec[:,:, 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
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
#NOTE THAT THE FLOW
#assuming flows is already ordered such that flows[0] = t and flows[n] = t - s
#NOTE THAT THIS FUNCTION CREATES SEQUENCES FOR THE FLOW AND ORDERING OF THE DATA MATTERS!
#IF T0 IS REFERENCING T THEN THEN WE ARE SEQUENCING FLOW VECTOR BACKWARDS FROM T TO T-N.
#IF T0 IS REFERENCING T-N THEN WE ARE SEQUENCING FLOW VECTOR FORWARD FROM T-N TO T
def stitchFlow(flows,mode='back'):
    n,h,w,d = flows.shape

    #SORRY I WAS TOO LAZY TO FIX THE DATA READING
    data = np.zeros((n,h,w,d))
    out1 = np.zeros((h,w,n,d))
    out2 = np.zeros((h,w,n,d))

    #loop through each flow and map the current pixels backwards in time
    for i,flow in enumerate(flows):
        t = (n-1) - i
        if i == 0:
            prv_map = np.dstack(np.meshgrid(np.arange(w),np.arange(h)))
            data[t] = flows[0]
            out2[:,:,i,:] = prv_map
            continue

        #MOVE THE PRVEVIOUS MAP TO THE CURRENT MAP
        #PLUS SIGN MOVE CUR_IMAGE BACKWARDS WITH CUR_FLOW
        if mode == 'back':
            cur_map = np.rint(prv_map + data[t+1]).astype(np.int64)
        else:
            cur_map = np.rint(prv_map - data[t+1]).astype(np.int64)

        #FIND THE NEW BOUNDRIES I.E SPOTS WHERE THE CURMAP FALL OFF THE IMAGE
        leftbound = np.logical_and(cur_map[:,:,0] >= 0,cur_map[:,:,1] >= 0)
        rightbound = np.logical_and(cur_map[:,:,1] < h,cur_map[:,:,0] < w )
        boundry = np.logical_and(leftbound,rightbound)

        #set data out of bounds to the previous
        data[t][np.logical_not(boundry)] = data[t+1][np.logical_not(boundry)]

        #WELL THIS IS PROBABLY MY MOST WELL MADE ALGORITHM...
        #THIS ONE SAYS TAKE YOUR BIPARTITE GRAPH DATA_t, FLOW AND CONNECT THEM USING CUR_MAP WHICH SETS DATA_ij to FLOW_ab by the
        #linear function C = A * w + B   where A is the list of column_id in CUR_MAP, B is the list of row_id in CUR_MAP, and C would be the
        #index of Flow_ij flattened. We then set DATA_ij to FLOW_ab if it is within the boundries of ij
        idx = (cur_map[boundry][:,1] * (w)) + cur_map[boundry][:,0]

        data[t][boundry][:,0] = flow[:,:,0].flatten()[idx]
        data[t][boundry][:,1] = flow[:,:,1].flatten()[idx]
        data[t][boundry] = np.dstack((flow[:,:,0].flatten()[idx],flow[:,:,1].flatten()[idx]))
        out2[:,:,i,:] = cur_map.copy()

        prv_map = cur_map.copy()

    for i in range(n):
        out1[:,:,i,:] = data[i,:,:,:]

    out2[:,:,:,0] = out2[:,:,:,0] / w
    out2[:,:,:,1] = out2[:,:,:,1] / h

    return out1,out2

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

    elif 'stitchflow' in sys.argv:

        HEIGHT = 436
        WIDTH = 1024
        DEPTH = 2

        filedir = sys.argv[2]
        topimg = 'data/flow/market2_frame_0049.png'
        #topimg = 'data/flow/cave4_frame_0049.png'

        if os.path.isdir(filedir):
            dirlist = os.listdir(filedir)
            dirlist.sort()
            dirlist.reverse()
            length = len(dirlist)
            flows = np.zeros((length,HEIGHT,WIDTH,DEPTH),dtype=np.float32)
            for i,f in enumerate(dirlist):
                flow = readflofile(os.path.join(filedir,f))
                h,w,d = flow.shape

                #no guarantees right?
                assert h == HEIGHT
                assert w == WIDTH
                assert d == DEPTH

                flows[i] = flow

            featuremap = stitchFlow(flows)
            imgtop = cv2.imread(topimg)
            video = genFlowSeq(imgtop,featuremap)

    elif 'stitchflow2' in sys.argv:

        HEIGHT = 436
        WIDTH = 1024
        DEPTH = 2

        filedir = sys.argv[2]
        topimg = 'data/flow/market2_frame_0001.png'
        #topimg = 'data/flow/cave4_frame_0049.png'

        if os.path.isdir(filedir):
            dirlist = os.listdir(filedir)
            dirlist.sort()
            length = len(dirlist)
            flows = np.zeros((length,HEIGHT,WIDTH,DEPTH),dtype=np.float32)
            for i,f in enumerate(dirlist):
                flow = readflofile(os.path.join(filedir,f))
                h,w,d = flow.shape

                #no guarantees right?
                assert h == HEIGHT
                assert w == WIDTH
                assert d == DEPTH

                flows[i] = flow

            featuremap = stitchFlow(flows)
            imgtop = cv2.imread(topimg)
            video = genFlowSeq(imgtop,featuremap)

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
        7. [stitchflow] [flow_dir]

        """)


