import cv2
import numpy as np
import imutils

def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255; # Some values seem to go above 255. However RGB channels has to be within 0-255

    return sobel

def findSignificantContours (img, sobel_8u):
    contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]]
        area = cv2.contourArea(contour)
        if area > tooSmall:
            cv2.drawContours(img, [contour], 0, (0,255,0),2, 3, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];

img = cv2.imread('../pictures/test4.jpg')

# resize of the image provides better precision
img = imutils.resize(img, width=600)

blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise

edgeImg = np.max( np.array([ edgedetect(blurred[:,:, 0]), edgedetect(blurred[:,:, 1]), edgedetect(blurred[:,:, 2]) ]), axis=0 )
mean = np.mean(edgeImg);
# Zero any value that is less than mean. This reduces a lot of noise.
edgeImg[edgeImg <= mean] = 0;


cv2.imwrite('../pictures/output/edges.jpg',edgeImg)

sobel_8u = np.asarray(edgeImg, np.uint8)

# Find contours
significant = findSignificantContours(img, sobel_8u)

#Mask
mask = edgeImg.copy()
mask[mask > 0] = 0
cv2.fillPoly(mask, significant, 255)

#Invert Mask
mask = np.logical_not(mask)

img[mask] = 0

cv2.imwrite('../pictures/output/final.jpg',img)