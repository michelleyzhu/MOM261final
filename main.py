import cv2
import numpy as np
import math

image = cv2.imread("ok.jpeg")
original = image.copy()

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255,
#     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow("Thresh", thresh)
blur = cv2.GaussianBlur(image,(5,5),0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# hsv_lower = np.array([156,60,0])
# hsv_upper = np.array([179,115,255])
hsv_lower = np.array([1,1,0])
hsv_upper = np.array([179,179,180])
mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

minimum_area = 10
average_cell_area = 650
connected_cell_area = 1000
cells = 0
total_cell_area = 0
for c in cnts:
    area = cv2.contourArea(c)
    # if area > minimum_area:
    cv2.drawContours(original, [c], -1, (36,255,12), 2)
    cv2.fillPoly(original, pts =[c], color=(255,255,255))
    total_cell_area += area
    if area > connected_cell_area:
        cells += math.ceil(area / average_cell_area)
    else:
        cells += 1
print('Cells: {}'.format(cells))
print('Cell area: {}'.format(total_cell_area))
cv2.imshow('close', close)
cv2.imshow('original', original)
cv2.waitKey()