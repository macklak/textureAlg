import numpy as np
# import matplotlib.pyplot as mpl
import cv2
# import math
import time
# image = cv2.imread("testIM.png")
# colors = cv2.imread("colors.png")
# image2=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
from config import V
#image[...,1]=0
#image[...,2]=0
# cv2.imshow('image',image)
# cv2.waitKey(0)
#print(np.shape(image))
# b,g,r = cv2.split(colors)
# #img = cv2.merge((b,g,r))
# cv2.imshow('blue', b)
# cv2.imshow('green', g)
# cv2.imshow('red', r)
# cv2.waitKey(0)
#cv2.imshow("image",image)
#cv2.waitKey(0)
# b,g,r=cv2.split(image)
# B=[]
# H=[]
# S=[]
# for i,value in enumerate(image):
#     B.append([])
#     H.append([])
#     S.append([])
#     for j,value in enumerate(image[i]):
#         u=min(r[i][j],g[i][j],b[i][j])
#         B[i].append(math.sqrt((r[i][j]**2+g[i][j]**2+b[i][j]**2)/3))
#         if r[i][j]==u:
#             H[i].append((2*math.pi/3)*(b[i][j]-u)/(g[i][j]+b[i][j]-2*u)+(math.pi)/3)
#         elif g[i][j]==u:
#             H[i].append( (2 * math.pi / 3)*(r[i][j] - u) / (b[i][j]+r[i][j] - 2 * u) + math.pi)
#         elif b[i][j]==u:
#             H[i].append( (2 * math.pi / 3)*(g[i][j] - u) / (g[i][j] + r[i][j] - 2 * u) + (5*math.pi)/3)
#         S[i].append(1-u/(r[i][j]+g[i][j]+b[i][j]))
######################################################################################################################
#код выше это моя реализация перехода от RGB(возможно тут BGR) к BHS(HSV вроде это одно и то же) и работает она неправильно
# так что думаю нет смысла ее переделывать если есть стандартная функция делающая тоже самое
########################################################################################################################
# H,S,V=cv2.split(image2)
t=[]
delta=1# та самая дельта для порога чувствительности
t0=time.time()
for indRow, row in enumerate(V):
    t.append([])
    #t0=time.time()
    for indCol, col in enumerate(V[indRow]):#выбираем точку
        for N in range(4):#масштаб
            t[indRow].append([])  # кубическая матрица под текстурные признаки
            L = 5
            vp = 0
            vm = 0
            qSum=0
            dMas=[]
            g=0
            d=0
            if indCol-L<0:#для границ
                L=indCol
            if indRow+L>len(V[row]):#тож для границ
                L=len(V[row])-indRow

            for i in range(indRow-L,indRow+L-1,N+1):#горизонтальный отрезок L
                dOld= d
                d=np.int16(V[i][indCol])-np.int16(V[i+1][indCol])#для знаков
                if np.sign(d) != np.sign(dOld) and abs(d)>delta:
                    g+=1
                qSum+=np.cos((i*np.pi)/(2*L))
                q=np.cos((i*np.pi)/(2*L))/qSum
                if d<0:
                    vm+=-d*q
                elif d>0:
                    vp+=d*q
            v=min(vp,vm)
            t[indRow][indCol].append(np.arctan(v*g))
            L = 5
            vp = 0
            vm = 0
            qSum = 0
            dMas = []
            g = 0
            d = 0


            if indRow-L<0:#для границ
                L=indRow
            elif indRow+L>len(V[col]):#тож для границ
                L=len(V[col])-indRow
            for i in range(indCol-L,indCol+L-2,N+1):# вертикаль
                dOld= d
                d=np.int16(V[indRow][i])-np.int16(V[indRow][i+1])
                if np.sign(d) != np.sign(dOld) and abs(d)>delta:
                    g+=1
                qSum+=np.cos((i*np.pi)/(2*L))
                q=np.cos((i*np.pi)/(2*L))/qSum
                if d<0:
                    vm+=-d*q
                elif d>0:
                    vp+=d*q
            v=min(vp,vm)
            t[indRow][indCol].append(np.arctan(v*g))

            L = 5
            vp = 0
            vm = 0
            qSum = 0
            dMas = []
            g = 0
            d = 0
            if indRow-L<0 or indCol-L<0:
                L=min(indRow,indCol)
            elif indRow+L>len(V[col]) or indCol+L>len(V[row]):
                L=len(V[max(len(row),len(col))])-max(indRow,indCol)
            for i in range(-L,L-2,N+1):#главная диагональ
                dOld= d
                d=np.int16(V[indRow+i][indCol+i])-np.int16(V[indRow+i+1][indCol+i+1])
                if np.sign(d) != np.sign(dOld) and abs(d)>delta:
                    g+=1
                qSum+=np.cos((i*np.pi)/(2*L))
                q=np.cos((i*np.pi)/(2*L))/qSum
                if d<0:
                    vm+=-d*q
                elif d>0:
                    vp+=d*q
            v=min(vp,vm)
            t[indRow][indCol].append(np.arctan(v*g))

            L = 5
            vp = 0
            vm = 0
            qSum = 0
            dMas = []
            g = 0
            d = 0
            if indRow - L < 0 or indCol - L < 0:
                L = min(indRow, indCol)
            elif indRow + L > len(V[row]) or indCol + L > len(V[row]):
                L=len(V[max(len(row),len(col))])-max(indRow,indCol)
            for i in range(- L, L  -1, N + 1):#побочная диагональ
                dOld = d
                d=np.int16(V[indRow-i][indCol+i])-np.int16(V[indRow-i+1][indCol+i+1])
                if np.sign(d) != np.sign(dOld) and abs(d) > delta:
                    g += 1
                qSum += np.cos((i * np.pi) / (2 * L))
                q = np.cos((i * np.pi) / (2 * L)) / qSum
                if d < 0:
                    vm += -d * q
                elif d > 0:
                    vp += d * q
            v = min(vp, vm)
            t[indRow][indCol].append(np.arctan(v * g))
t1=time.time()-t0
print("циклы для 1 направления" , t1)






