#!/usr/bin/env python
# coding: utf-8

# In[131]:


import cv2
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt
def imageinput(filename):
    original=filename
    img = cv.imread(original,0)
    img = cv2.GaussianBlur(img,(5,5),0)
    return img


# In[148]:


def imagethresholding(filename):
    file=filename
    img=imageinput(file)
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,                cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,                cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return th1


# In[133]:


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
                            [(30, 700),  (1250,700),(750,200),(550, 200)]
                        ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment


# In[134]:


def canny_detection(filename):
    img=imageinput(filename=filename)
    canny1 = cv2.Canny(img,50,150,apertureSize = 3)
    seg1=do_segment(canny1)
    images = [seg1,canny1]
    titles = ['Canny Segmentaion', 'Canny edge']
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return seg1


# In[135]:


def main(filename):
    file=filename
    dst1=canny_detection(file)
    src = cv.imread(filename=filename)
    src2=np.copy(src)
    cdst = dst1

    linesP = cv.HoughLinesP(dst1, 1, np.pi / 120, 100,None, 120, 40)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    cv.imshow("Source", src)
    cv.imshow("Detected Lines  - Probabilistic Line Transform", src2)
    cv.waitKey()
    return 0
    


# In[146]:


def canny_binary(filename):
    file=filename
    th1=imagethresholding(file)
    edges1 = cv2.Canny(th1,50,130,apertureSize = 3)
    edges3 = cv2.Canny(th3,50,130,apertureSize = 3)
    cb1=do_segment(edges1)
    images = [cb1, th1, cb2, th3]
    titles = ['Canny Binary', 'Binary Thresholding ']
    for i in range(2):
        plt.subplot(1,2,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return cb1


# In[143]:


def main_binary(filename):
    file=filename
    dst1=canny_binary(file)
    src = cv.imread(file)
    src2=np.copy(src)
    cdst = dst1
    linesP = cv.HoughLinesP(dst1, 1, np.pi / 180, 80,None, 150, 50)
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(src2, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    cv.imshow("Source", src)
    cv.imshow("Detected Lines (binary) - Probabilistic Line Transform", src2)
    cv.waitKey()
    return 0


# In[149]:


if __name__ == "__main__":
    original = 'b3.jpg'
    method = ''
    if method == 'gaussian':
        main(filename=original)
    else:
        main_binary(original)


# In[ ]:




