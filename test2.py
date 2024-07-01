import numpy as np
import matplotlib.pyplot as mpl
import time
import cv2
import multiprocessing as mp



def timeit(func):
    def _wrapper(*args,**kwargs):
        t0=time.time()
        func(*args,**kwargs)
        t1=time.time()-t0
        print(t1)
    return _wrapper

#@timeit
def steps(A):
    with mp.Pool(2) as p:
        arrV=[p.map(findT, np.stack([np.roll(A, i*j, axis=0) for i in range(-5, 5)]for j in range(1,5))) ]
        arrV = np.squeeze(arrV, axis=0)
    print(np.shape(arrV))
    arrW= np.stack([findT(np.stack([np.roll(A, i * j, axis=1) for i in range(-5, 5)])) for j in range(1, 5)])
    arrPlus = np.stack([findT(np.stack([np.roll(np.roll(A, i*j, axis=0),i*j,axis =1) for i in range(-5, 5)])) for j in range(1, 5)])
    arrMin = np.stack([findT(np.stack([np.roll(np.roll(A, i*j, axis=0),-1*i*j,axis =1) for i in range(-5, 5)])) for j in range(1, 5)])
    arrFul=np.concatenate([arrV,arrW,arrPlus,arrMin],axis=0)
    return arrFul



def findT(arr,L=5):
    qSum = np.cos((arr[0] * np.pi) / (2 * L))
    vp = np.zeros(np.shape(arr[0]))
    vm = np.zeros(np.shape(arr[0]))
    zeros = np.zeros(np.shape(arr[0]))
    for i in range(9):
        d=arr[i]-arr[i+1]
        print(d)
        q = np.cos((i * np.pi) / (2 * L)) / (qSum+0.000001)
        s = np.sign(d)
        g = np.sum((s[1:]-s[:-1]) !=0,axis=0)
        qSum += np.cos((arr[i]*np.pi)/(2*L))
        vm += -1*(np.where(d<0,d*q,zeros))
        vp += np.where(d > 0, d*q,  zeros)
    v=np.minimum(vp,vm)
    t=np.arctan(v*g)
    return t
if __name__ =="__main__":
    image = cv2.imread("testIM.png")
    image2=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H,S,V=cv2.split(image2)
    #mpl.imshow(np.roll(V,100,axis=0),cmap='gray')
    #arr= np.stack([np.roll(V,i*4, axis=0) for i in range(-5,5)])
    #arr=arr.astype("float")
    # print(np.shape(V))

    text4=steps(V/255)
    mpl.imshow(text4[0], cmap='gray')

