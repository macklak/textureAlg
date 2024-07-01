import numpy as np
import matplotlib.pyplot as mpl
import time
import cv2
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__() # just run the init of parent class (nn.Module)
        image = cv2.imread("image.jpg")
        image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(image2)
        self.conv1 = nn.Conv2d(1, 32, 5) # вход 1 изображение, выход 32 канала, ядро свертки размером 5x5
        self.conv2 = nn.Conv2d(32, 64, 5) # вход 32, так как выход первого слоя 32. Выход будет 64 канала, ядро свертки размером 5x5
        self.conv3 = nn.Conv2d(64, 128, 5)
        x = torch.randn(50,50).view(-1,1,50,50)
        self._to_linear = None
        self.convs(V)
        self.fc1 = nn.Linear(self._to_linear, 512) #выпрямление.
        self.fc2 = nn.Linear(512, 2) # 512 вход, 2 выход так как у нас два класса (собаки и кошки).
    def convs(self, x):
        # max pooling over 2x2
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Это наш выходной слой. Функции активации тут нет.
        return F.softmax(x, dim=1)












if __name__ =="__main__":

    net = Net()
    print(net)