import cv2
import numpy as np
import os

# img = 'imgs/neg1_a.JPG'
sz = 1300
files = sorted(os.listdir("imgs"))
for img in files:
    if "X" not in img:
        print(img,": ",end='')
        image = cv2.imread("imgs/" + img)
        h,w = image.shape[0],image.shape[1]
        mask = np.zeros((h,w), np.uint8)
        cY,cX = h//2,w//2
        rad = (int)(cX * 0.7)
        circle_img = cv2.circle(mask,(cY,cX),rad,(255,255,255),thickness=-1)
        masked_data = cv2.bitwise_and(image, image, mask=circle_img)
        cropped = masked_data[cY-rad:cY+rad,cX-rad:cX+rad]
        resized = cv2.resize(cropped,(sz,sz))
        # resizedOrig = cv2.resize(image[cY-rad:cY+rad,cX-rad:cX+rad],(sz,sz))
        gray = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
        thresh = (gray > 2*np.mean(gray)).astype('uint8')
        print(np.count_nonzero(thresh) * 100/ (sz**2))

        display = 255 * thresh
        stacked = np.hstack((np.dstack((display,display,display)),resized))
        cv2.imshow("thresh",stacked)

        cv2.waitKey(0)