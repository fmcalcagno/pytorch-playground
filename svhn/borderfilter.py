import os
import cv2
import glob
import dataset

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage

folder_data = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\data"


folder_input = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages"

images = [(cv2.imread(file),file) for file in glob.glob("C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages\\*.png")]



#kernel = np.ones((5,5),np.float32)/25

#9
kernel_hor_9 = np.array([[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1],[1,1, 1,0,0,0,1,1,1]])/44
kernel_vert_9= np.array([[1,1, 1,1,1,1,1,1,1],[1,1, 1,1,1,1,1,1,1],[1,1, 1,1,1,1,1,1,1],[0,0, 0,0,0,0,0,0,0],[0,0, 0,0,0,0,0,0,0],[0,0, 0,0,0,0,0,0,0],[1,1, 1,1,1,1,1,1,1],[1,1, 1,1,1,1,1,1,1],[1,1, 1,1,1,1,1,1,1]])/44

#7
kernel_hor_7 = np.array([[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1],[1,1,0,0,0,1,1]])/28
kernel_vert_7= np.array([[1,1, 1,1,1,1,1],[1,1, 1,1,1,1,1],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1]])/28

#5
kernel_hor_5 = np.array([[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1],[1,0,-1,0,1]])/5
kernel_vert_5= np.array([[1,1, 1,1,1],[0,0,0,0,0],[-1,-1,-1,-1,-1],[0,0,0,0,0],[1,1, 1,1,1]])/5
#3
kernel_hor_3 = np.array([[1,0,1],[1,0,1],[1,0,1]])/5
kernel_vert_3= np.array([[1,1,1],[0,0,0],[1,1,1]])/5



for img,file in  images:
    img = cv2.resize(img, (32, 32)) 
    dst_vert_3 = cv2.filter2D(img,-1,kernel_vert_3)
    dst_hor_3 = cv2.filter2D(img,-1,kernel_hor_3)
    dst_vert_hor_3 =cv2.filter2D(cv2.filter2D(img,-1,kernel_vert_3),-1,kernel_hor_3)

    dst_vert_5 = cv2.filter2D(img,-1,kernel_vert_5)
    dst_hor_5 = cv2.filter2D(img,-1,kernel_hor_5)
    dst_vert_hor_5 =cv2.filter2D(cv2.filter2D(img,-1,kernel_vert_5),-1,kernel_hor_5)

    dst_vert_7 = cv2.filter2D(img,-1,kernel_vert_7)
    dst_hor_7 = cv2.filter2D(img,-1,kernel_hor_7)
    dst_vert_hor_7 =cv2.filter2D(cv2.filter2D(img,-1,kernel_vert_7),-1,kernel_hor_7)

    dst_vert_9 = cv2.filter2D(img,-1,kernel_vert_9)
    dst_hor_9 = cv2.filter2D(img,-1,kernel_hor_9)
    dst_vert_hor_9 =cv2.filter2D(cv2.filter2D(img,-1,kernel_vert_9),-1,kernel_hor_9)

    
    plt.subplot(441),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(442),plt.imshow(dst_vert_3),plt.title('3 Vertical Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(443),plt.imshow(dst_hor_3),plt.title('3 Horizontal Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(444),plt.imshow(dst_vert_hor_3),plt.title('3 Hor/Vert Filter')
    plt.xticks([]), plt.yticks([])

    plt.subplot(445),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(446),plt.imshow(dst_vert_5),plt.title('5 Vertical Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(447),plt.imshow(dst_hor_5),plt.title('5 Horizontal Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(448),plt.imshow(dst_vert_hor_5),plt.title('5 Hor/Vert Filter')
    plt.xticks([]), plt.yticks([])
   
    plt.subplot(449),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,10),plt.imshow(dst_vert_7),plt.title('7 Vertical Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,11),plt.imshow(dst_hor_7),plt.title('7 Horizontal Filter')
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(4,4,12),plt.imshow(dst_vert_hor_7),plt.title('7 Hor/Vert Filter')
    plt.xticks([]), plt.yticks([]) 
         

    plt.subplot(4,4,13),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,14),plt.imshow(dst_vert_9),plt.title('9 Vertical Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(4,4,15),plt.imshow(dst_hor_9),plt.title('9 Horizontal Filter')
    plt.xticks([]), plt.yticks([]) 
    plt.subplot(4,4,16),plt.imshow(dst_vert_hor_9),plt.title('9 Hor/Vert Filter')
    plt.xticks([]), plt.yticks([]) 
     

    plt.show()
