import numpy as np
# import matplotlib.pyplot as mpl
import time
# import cv2
# import multiprocessing as mp
# from numba import njit
from config import V
def steps(func):
     def _wrapper(*args,**kwargs):
         arrV=np.stack([func(np.stack([np.roll(V/255, i*j, axis=0) for i in range(-5, 5)])) for j in range(1,5)])
         arrW= np.stack([func(np.stack([np.roll(V / 255, i * j, axis=1) for i in range(-5, 5)])) for j in range(1, 5)])
         arrPlus = np.stack([func(np.stack([np.roll(np.roll(V / 255, i*j, axis=0),i*j,axis =1) for i in range(-5, 5)])) for j in range(1, 5)])
         arrMin = np.stack([func(np.stack([np.roll(np.roll(V / 255, i*j, axis=0),-1*i*j,axis =1) for i in range(-5, 5)])) for j in range(1, 5)])
         arrFul=np.concatenate([arrV,arrW,arrPlus,arrMin],axis=0)
         #print(np.shape(arrFul))
         return arrFul
     return _wrapper

def timeit(func):
    def _wrapper(*args,**kwargs):
        t0=time.time()
        func(*args,**kwargs)
        t1=time.time()-t0

        print('массивы numpy',t1)
    return _wrapper
@timeit
@steps
#@njit(parallel=True)
def findT(arr,L=5):
    qSum = np.cos((arr[0] * np.pi) / (2 * L))
    vp = np.zeros(np.shape(arr[0]))
    vm = np.zeros(np.shape(arr[0]))
    zeros = np.zeros(np.shape(arr[0]))
    for i in range(9):
        d=arr[i]-arr[i+1]
        q = np.cos((i * np.pi) / (2 * L)) / (qSum+0.000001)
        s = np.sign(d)
        g = np.sum((s[1:]-s[:-1]) !=0,axis=0)
        qSum += np.cos((arr[i]*np.pi)/(2*L))
        vm += -1*(np.where(d<0,d*q,zeros))
        vp += np.where(d > 0, d*q,  zeros)
    v=np.minimum(vp,vm)
    t=np.arctan(v*g)
    return t

# image = cv2.imread("testIM.png")
# image2=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# H,S,V=cv2.split(image2)



#mpl.imshow(np.roll(V,100,axis=0),cmap='gray')
#arr= np.stack([np.roll(V,i*4, axis=0) for i in range(-5,5)])
#arr=arr.astype("float")
# print(np.shape(V))
text4=findT(V/255)

# mpl.figure(figsize=(10,10))
# mpl.imshow(text4[0],cmap='gray')
# for i, text in enumerate(text4):
#     print(text.min(),text.max())
# text=text4[0]
# text -= text4.min()
# text /= text4.max()
# text = np.uint8(text4[0]*255)
# cv2.imwrite(f'textureNEW{0}.png', text)
#     mpl.subplot(4, 4, i+1)
#     mpl.imshow(text,cmap='gray')
#     mpl.show()
#np.save('mean',text)
#mpl.savefig('mean.png') #границы торчат
#cv2.imwrite('mean.png',text*255)
#print(np.shape(text4))
# for i in range(4):
#     mpl.subplot(2,2,i+1)
#     mpl.imshow(text4[i],cmap='gray')
# mpl.show()
#mpl.imshow(findT(),cmap='gray')

