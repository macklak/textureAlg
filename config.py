import os
import cv2
# os.system("pip install numpy")
# os.system("pip install opencv-python")
# os.system("pip install torch")
im = "testIM.png"
image = cv2.imread(im)
image2=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
H,S,V=cv2.split(image2)