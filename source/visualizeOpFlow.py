import sys
import os
import cvbase as cvb
import numpy as np

if len(sys.argv) == 2 and os.path.exists(sys.argv[1]):
    cvb.show_flow(sys.argv[1])

    #flow = np.random.rand(100,100,2).astype(np.float32)

    #cvb.show_flow(flow)
else:

    print('ERROR with args please call as follows')
    print('python visualizeOpFlow.py [file.flo]')
