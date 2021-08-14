# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 23:02:09 2021

@author: Azul
"""
import imageio
from scipy import ndimage
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#%%
path="calibracion (1).mp4"
cap=cv.VideoCapture(path)

cap.set(1, 0)
ret, frame=cap.read()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 
#%%
plt.imshow(gray)
R=gray[110:400, 300:400]

R_rot=ndimage.rotate(R, 2)

R_r=R_rot[34:270, 15:100]

plt.imshow(R_r)
#%%
suma=np.sum(R_r, axis=1)
suma2=np.sum(R_r, axis=0)

plt.figure(3), plt.clf()
plt.plot(suma)

x = plt.ginput(10)

diferencias=[x[i][0]-x[i-1][0] for i in range(1, len(x))] 
#%%
milimetros_5=np.mean(diferencias)
error=np.std(diferencias)/np.sqrt(len(diferencias))

print("milimetros=", milimetros_5/5, "error=", error/5)

