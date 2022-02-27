# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 13:41:36 2021

@author: abhil
"""
import cv2
import numpy as np
import os
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    transition = np.empty((1,256), dtype=np.uint8)
    for elem in range(256):
        transition[0,elem] = np.clip(((elem / 255.0)** gamma) * 255.0, 0, 255)
    result = cv2.LUT(image, transition)
    return result

file_location=os.getcwd()  
video=cv2.VideoCapture(file_location+"/data_2/challenge_video.mp4")

out = cv2.VideoWriter('challenge_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1280,720))


K =np.array( [[  1.15422732e+03,   0.00000000e+00 ,  6.71627794e+02],
 [  0.00000000e+00 ,  1.14818221e+03 ,  3.86046312e+02],
 [  0.00000000e+00 ,  0.00000000e+00  , 1.00000000e+00]])
dist = np.array([[ -2.42565104e-01 , -4.77893070e-02 , -1.31388084e-03 , -8.79107779e-05,
    2.20573263e-02]])

src_pts=np.array([[83,316],[1242,316],[600,111],[755,111]])

dest_pts=np.array([[0,600],[600,600],[0,0],[600,0]])


upper_y=np.array([35,190,255])
lower_y=np.array([0,0,50])

upper_w=np.array([250,250,250])
lower_w=np.array([0,180,0])


while True:
    
    ret,frame=video.read()
    if frame is None:
        break
    #preprocessing
    orig=np.zeros(frame.shape,dtype=np.uint8)
    dst=cv2.undistort(frame, K,dist,None, K)
    cv2.imshow("frame",frame)
    cv2.imshow("dst",dst)
    #cv2.imshow("dst",dst)
    blur=cv2.medianBlur(dst,ksize=3)
    #blur=cv2.GaussianBlur(dst,ksize=(3,3), sigmaX=0)
    #blur=dst
    b=cv2.resize(blur,(380,240))
    cv2.imshow("b",b)
    m,n,c=blur.shape
    blur_s=blur[int(720/2):][:][:]
    blur_untouched=blur[0:int(720/2)][:][:]
    cv2.imshow("blur_u",blur_untouched)
    blur=blur_s
    #cv2.imwrite("blur1.jpg",blur)
    blur_hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HLS)
    
    #Changing the intensity or image channel values acccording to the lighting condition 
    
    
    th = np.median(frame[int(frame.shape[0]/2):,:,:])
    if th>130:
        
        blur = adjust_gamma(blur,gamma=1.0)

    elif th<60:
        
        hsv_v = blur_hsv[:,:,1]
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))#increased values may cause noise
        cl1 = clahe.apply(hsv_v)
   
        gamma = 0.65
        cl1= adjust_gamma(cl1, gamma=gamma)
        
        blur_hsv[:,:,1] = cl1
    
        


    
    
    

    #calculating homography
    h,ret=cv2.findHomography(src_pts,dest_pts)
    #print(h)
    inv_wrap=cv2.warpPerspective(blur,h,(600,600))
    cv2.imshow("inv_wrap",inv_wrap)
    
    
    #creating mask for yellow and white lanes
    inv_hsv=cv2.cvtColor(inv_wrap,cv2.COLOR_BGR2HLS)
    mask_b=cv2.inRange(blur_hsv,lower_y,upper_y)
    #mask_orignal=cv2.cvtColor(mask,cv2.COLOR_HSV2RGB)
    make=cv2.bitwise_and(blur_hsv,blur_hsv,mask=mask_b)
    make=cv2.cvtColor(make,cv2.COLOR_HLS2BGR)
    mask_w_b=cv2.inRange(blur_hsv,lower_w,upper_w)
    #mask_orignal=cv2.cvtColor(mask,cv2.COLOR_HSV2RGB)
    make_w=cv2.bitwise_and(blur_hsv,blur_hsv,mask=mask_w_b)
    make_w=cv2.cvtColor(make_w,cv2.COLOR_HLS2BGR)
    original_b=cv2.bitwise_or(mask_w_b,mask_b)
    original=cv2.bitwise_or(make_w,make)
    cv2.imshow("w",original)
    cv2.imshow("mask_inverse_w",make_w)
    #cv2.imshow("mask",mask)
    cv2.imshow("mask_inverse",make)
    cv2.imshow("blur",blur)
    
    # Changing perspective to top for mask
    
    
    
    wrap_mask=cv2.warpPerspective(original,h,(600,600))
    mask_b=cv2.warpPerspective(original_b,h,(600,600))
    
    #Thresholding into binary image
    ret,binary=cv2.threshold(mask_b,41,1,cv2.THRESH_BINARY)
    cv2.imshow("inv_mask1+2",wrap_mask)
    gray=cv2.cvtColor(wrap_mask,cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray",gray)
    #binary=cv2.threshold(gray,10,255,)
    cv2.imshow("binary",mask_b)
    
    
    #Calculating histogram and getting x coordinate of the lanes 
    sum_horiz=np.sum(binary,axis=0)
    mid=int(len(sum_horiz)/2)
    line1=np.argmax(sum_horiz[:mid])
    #print("x",np.argmax(sum_horiz[:mid]))
    line2=np.argmax(sum_horiz[mid:])+mid
    
    
    #Creating window and getting the position of all points of the laned
    
    left_line =binary[:,line1-50:line1+50]
    right_line=binary[:,line2-50:line2+50]
    lane_center=(line1+(line2-line1)*0.5)
    left_pts=np.where(left_line==1)
    right_pts=np.where(right_line==1)
    
    #Curve Fitting the points and getting equation of lines
    #and in turn getting the all coordinates of the lane and plotting
    
    fit_y_l=np.polyfit(left_pts[0],left_pts[1]+line1-50,2)
    fit_y_r=np.polyfit(right_pts[0],right_pts[1]+line2-50,2)
    
    x=np.arange(len(binary))
    XX=np.stack((x**2,x,np.ones(len(binary))))
    points_to_plot_l=(fit_y_l @ XX)
    points_to_plot_r=np.dot(fit_y_r,XX)
    points_to_plot_l=points_to_plot_l.astype(np.uint32)
    points_to_plot_r=points_to_plot_r.astype(np.uint32)
    pts_l=np.asarray([points_to_plot_l,x]).T.astype(np.int32)
    pts_r=np.asarray([points_to_plot_r,x]).T.astype(np.int32) 
    impose=np.zeros((600,600)).astype(np.uint8)
    cv2.polylines(inv_wrap,[pts_l],False,(255,255,255))
    cv2.polylines(inv_wrap,[pts_r],False,(255,255,255))
   
    #Calculating gradient for direction of turn
    
    avg=((pts_r+pts_l)/2).astype(np.int32)
    grad_value=np.gradient(avg[:int(avg.shape[0]/2),0],2)
    grad_avg=(np.average(grad_value))
    #print("grad_avg",grad_avg)
    
    #Filling the lane with blue colour and inverse wrapping on the orignal image
    
    pts_r[:,0]=pts_r[::-1,0]
    pts_r[:,1]=pts_r[::-1,1]
    pts=np.vstack((pts_l,pts_r))
    cv2.fillPoly(inv_wrap,[pts],(140,140,20))
    cv2.polylines(inv_wrap,[avg],False,(255,255,255))
    inverse_wrap=cv2.warpPerspective(inv_wrap,np.linalg.inv(h),(blur.shape[1],blur.shape[0]))
    cv2.imshow("curve",impose)    
    #print("arg1",line1,line2)
    cv2.imshow("inverse_wrap",inverse_wrap)
    #print("frame.shape",frame.shape)
    output=cv2.addWeighted(blur,1,inverse_wrap,0.5,0)
    cv2.imshow("output",output)
    orig[int(720/2):][:][:]=output
    orig[:int(720/2)][:][:]=blur_untouched
    
    
    # direction from gradient
    
    if grad_avg <=-0.01:
        print("right")
        cv2.putText(orig,"TURN : right",org=(1050,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(140,140,20))
    elif grad_avg>0.1:
        print("left")
        cv2.putText(orig,"TURN : left",org=(1050,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(140,140,20))
    elif grad_avg>-0.01 and grad_avg<=0.1:
        print("straight")
        cv2.putText(orig,"TURN : straight",org=(1050,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(140,140,20))
    cv2.imshow("orig",orig)
    out.write(orig)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
video.release()
out.release()
cv2.destroyAllWindows()
            

