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

images = [(cv2.imread(file,1),file) for file in glob.glob("C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\testingimages\\*.png")]


def adjust_gamma(image, gamma=1.0):
    	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def cond_vert(filter_im):

    filter_im = cv2.cvtColor(filter_im, cv2.COLOR_BGR2GRAY)
    im_np = np.asarray(filter_im)

    first3=np.average(im_np[:,0:1])
    middle3=np.average(im_np[:,2:3])
    last3=np.average(im_np[:,4:5])

    
    return True if first3< middle3*0.84 and last3< middle3*0.84  else False

 
def cond_hor(filter_im):
    filter_im = cv2.cvtColor(filter_im, cv2.COLOR_BGR2GRAY)
    im_np = np.asarray(filter_im)
    first3=np.average(im_np[0:1,:])
    middle3=np.average(im_np[2:3,:])
    last3=np.average(im_np[4:5,:])

    return True if first3< middle3*0.84 and last3< middle3*0.84  else False

       


def holesearch(img):
 
    height, width = img.shape[:2]
    filter_out= np.zeros([height,width,3])
    filter_out2= np.zeros([height,width,3])
    filterh,filterw= 3,6
    filterh2,filterw2= 6,3
    im_out= np.asarray(img)
    
    positions_vert,positions_hor=[],[]
    

    for i in range( height- filterh+1):
        for j in range(width-filterw+1):
            filter_im = img [i:i+filterh-1,j:j+filterw]
            if cond_vert(filter_im):
                filter_out[i:i+filterh-1,j+1:j+4]=(255,255,255)
                positions_vert.append((i,j))
            filter_im2 = img [i:i+filterh2,j:j+filterw2-1]
            if cond_hor(filter_im2):
                filter_out2[i+1:i+4,j:j+filterw2]=(255,255,255)
                positions_hor.append((i,j))

    for i,j in positions_vert:
        im_out[i:i+filterh-1,j+1:j+4]= img[i:i+filterh-1,j:j+1]
    for i,j in positions_hor:
        im_out[i+1:i+4,j:j+filterw2]= img[i:i+1,j:j+filterw2]
                #Fix image
    return filter_out,filter_out2,im_out
            


for img,file in  images:
    img = cv2.resize(img, (32, 32)) 
    #blur = cv2.GaussianBlur(img,(5,5),0)
    #median = cv2.medianBlur(img,5)t

    #hole search
    plt.subplot(151),plt.imshow(img),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    hole,hole2,im_out = holesearch(img)
    adjusted = adjust_gamma(img, gamma=0.8)
    
    
    
    plt.subplot(152),plt.imshow(hole),plt.title('Hole Filter')
    plt.xticks([]), plt.yticks([])
    plt.subplot(153),plt.imshow(hole2),plt.title('Hole Filter 2')
    plt.xticks([]), plt.yticks([])
    plt.subplot(154),plt.imshow(im_out),plt.title('Fixed Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(155),plt.imshow(adjusted),plt.title('Fixed Image Contrast')
    plt.xticks([]), plt.yticks([])


    plt.show()
    base=os.path.basename(file)
    cv2.imwrite(".\\testingimages\\modified-" + base, adjusted)
    #cv2.waitKey(0)                 # Waits forever for user to press any key
    #cv2.destroyAllWindows() 

    