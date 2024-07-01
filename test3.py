import numpy as np
import matplotlib.pyplot as mpl
import time
import cv2
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from scipy.signal import convolve


def timeit(func):
    def _wrapper(*args,**kwargs):
        t0=time.time()
        func(*args,**kwargs)
        t1=time.time()-t0
        print(t1)
    return _wrapper

#@timeit
def steps(A,ker):
    cos = np.zeros((9,9))
    kerS = np.zeros((3, 3))
    kerS[1][1] += 1/2
    kerS[0][1] -= 1/2
    for i in range(9):
        cos[i][4]=np.cos((np.pi*i)/8)
    arrV = np.stack([findT(A,ker[j-1],cos,kerS)for j in range(1,5)])
    return arrV

def findT(arr,ker,kerCos,kerS,L=5):
    zeros = np.zeros(np.shape(arr))
    d = convolve(arr, ker,mode="same")
    dq=convolve(d, kerCos,mode="same")
    s = np.sign(d)
    g = convolve(s, kerS,mode="same")
    vm = -1 * (np.where(d < 0, dq, zeros))
    vp = np.where(d > 0, dq, zeros)
    v=np.minimum(vp,vm)
    t=np.arctan(v*g)
    return t

if __name__ =="__main__":
    image = cv2.imread("testIM.png")
    image2=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(image2)
    kerd= np.zeros((4,9,9))
    for i in range(4):
        kerd[i][4][4] += 1
        kerd[i][i][4] -= 1

    print(kerd)
    v=steps(V/255,kerd)
    #mpl.imshow(v[2], cmap='gray')