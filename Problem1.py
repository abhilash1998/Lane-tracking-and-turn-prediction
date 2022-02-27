# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
video=cv2.VideoCapture(r'C:/Users/abhil/OneDrive/Desktop/ENPM673/Project2/Night Drive - 2689.mp4')
out = cv2.VideoWriter('hist_equal.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (640,480))

def adjust_gamma(image, inv_gamma=1.0):
    
    """
    Function takes the image and reduces or increase  its lightning

    Parameters
    ----------
    image : np.array
        Enhanced  image by histogram
    gamma : TYPE, optional
        DESCRIPTION. gamma factor.

    Returns
    -------
    result : np.array
        returns resultant image

    """
	
    transition = np.empty((1,256), dtype=np.uint8)
    for elem in range(256):
        transition[0,elem] = np.clip(((elem / 255.0)** inv_gamma) * 255.0, 0, 255)
    result = cv2.LUT(image, transition)
    return result
def histogram(image):
    """
    

    Parameters
    ----------
    image : np.array
        input image/frame of the video

    Returns
    -------
    hist : np.array
        histogram of colour windows

    """
    hist=np.zeros(256)
    #m,n=image.shape
    
    m,n=np.unique(image,return_counts=True)
    hist[m]=n
    return hist
def cumsum(hist):
    """
    Returns the cummulitive sum of the histogram    
    
    Parameters
    ----------
    hist : np.array
        histogram of frame/orignal image.

    Returns
    -------
    cum_hist : np.array
        cummulative sum of histogram.

    """
    cum_hist=np.cumsum(hist)
    return cum_hist

def mapped_image(cum_hist):
    """
    
    Calculates the CDF and returns mapping array
    Parameters
    ----------
    cum_hist : np.array
        cummulative sum of histogram.


    Returns
    -------
    mapping : np.array
        low light enhanced array 

    """    
    
    mapping = np.maximum(0, np.round((255*cum_hist)/(640*480))-1)
    
    return mapping
def retriving_image(mapping,img):
    """
    
    Changing image colour values with
    enhanced colour values

    Parameters
    ----------
    mapping : np.array
        low light enhanced array .
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    newimg : np.array
        Final improved ligtning image.

    """
    newimg=mapping[img]
    newimg=newimg.astype(np.uint8)
    return newimg
    
m=640
n=480
while True:
    ret,frame=video.read()
    if frame is None:
        break
    frame=cv2.resize(frame, (640,480))
    frame=cv2.GaussianBlur(frame,ksize=(3,3),sigmaX=0)
    cv2.imshow("orignal_output",frame)
    image_change=cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    image_change_f=image_change.copy()
    
    hist=histogram(image_change[:,:,0])
    cum_hist=cumsum(hist)
    image_map=mapped_image(cum_hist)
    image_change_f[:,:,0]=retriving_image(image_map,image_change[:,:,0])
    image_change_f[:,:,0]=adjust_gamma(image_change_f[:,:,0],15)
    image_f=cv2.cvtColor(image_change_f, cv2.COLOR_YCrCb2RGB) 
    
    cv2.imshow("my_output",image_f)
    out.write(image_f)

    if cv2.waitKey(0) & 0xFF==ord('q'):
       break
   
out.release() 
video.release()
cv2.destroyAllWindows()
