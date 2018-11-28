import cv2
import glob


folder_data = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\data"

folder_input = "C:\\Users\\fcalcagno\\Documents\\pytorch-playground\\svhn\\siftimages"


images = [(cv2.imread(file),file) for file in glob.glob(folder_input + "\\*.png")]

sift = cv2.xfeatures2d.SIFT_create()

for im,file in images:
   
    gray= cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    kp = sift.detect(gray,None)
    im_sift=cv2.drawKeypoints(gray,kp)

    cv2.imshow('image',im)
    cv2.imshow('image',im_sift)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
