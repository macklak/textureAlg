from torch import Tensor
from torch.nn.functional import conv2d
# from string import Template
import cv2 as cv
import numpy as np
import torch
import time
from config import im
filename = im
from copy import copy
#S = 4
im = Tensor(cv.imread(filename, cv.IMREAD_GRAYSCALE))[None,None,...]

L = 5
#q = Tensor([np.cos(i*np.pi/(2*L)) for i in range(-L,L+1)])
q = np.zeros((4,1,2*L+1,2*L+1))
summ = np.zeros((4,1,2*L+1,2*L+1))
for i in range(2*L+1):
  q[0,0][i][L]=np.cos(i*np.pi/2*L)
  q[1,0][L][i]=np.cos(i*np.pi/2*L)
  q[2,0][2*L-i][i]=np.cos(i*np.pi/2*L)
  q[3,0][i][i]=np.cos(i*np.pi/2*L)
  summ[0,0][i][L]=1
  summ[1,0][L][i]=1
  summ[2,0][2*L-i][i]=1
  summ[3,0][i][i]=1
#q = np.concatenate((q,q.copy(),q.copy(),q.copy()),axis = 0)
summ = Tensor(np.array(summ))
q = (q/(q.sum()*4))
q = Tensor(np.array(q))

gr = Tensor([[[[0],[-1],[0]],[[0],[1],[0]],[[0],[0],[0]]],
            [[[0],[0],[0]],[[0],[1],[-1]],[[0],[0],[0]]],
            [[[0],[0],[-1]],[[0],[1],[0]],[[0],[0],[0]]],
            [[[0],[0],[0]],[[0],[1],[0]],[[0],[0],[-1]]]]).reshape(4,1,3,3)
t1=[]
t0=time.time()
for S in range(1,5):
    d = conv2d(im/255, gr, padding='same',dilation=S)
    dp = torch.zeros_like(d)
    dm = torch.zeros_like(d)
    dp[d>0] = d[d>0]
    dm[d<0] = - d[d<0]

    vnn = torch.cat([dp,dm])
    v = conv2d(vnn, q, padding='same',dilation=S,groups=4)
    #vm = conv2d(dm, q, padding='same',dilation=S,groups=4)
    v = torch.minimum(v[0],v[1]) #,keepdims=True)
    v = v[None,:,:,:]

    g = conv2d(
        conv2d(torch.sign(d),gr,padding='same',dilation=S,groups=4),
        summ, padding='same', groups=4
    )

    t = torch.arctan(v*g)
    t1.append(t)
t1=time.time()-t0
print("свертка без предобработки" , t1)