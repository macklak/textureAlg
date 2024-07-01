import numpy as np
import cv2
im = np.zeros((256,256),dtype=np.uint8)

grid = np.mgrid[0:128,0:128]
print(grid.shape)
im[:128,:128][((grid[0]+grid[1])%4)//2==0]=255
im[128:,:128][((grid[0])%4)//2==0]=255
im[:128,128:][((grid[1])%4)//2==0]=255
im[128:,128:][((grid[0]-grid[1])%4)//2==0]=255


cv2.imwrite("testIM.png",im)