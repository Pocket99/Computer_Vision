# import other necessary libaries
from utils import create_line, create_mask
import torch
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# load the input image
img = cv.imread('road.jpg',0)

# run Canny edge detector to find edge points
edges = cv.Canny(img,100,200)


# create a mask for ROI by calling create_mask
mask = create_mask(edges.shape[0], edges.shape[1])

# extract edge points in ROI by multipling edge map with the mask
edge_points = edges * mask

# perform Hough transform
# initialize accumulator
accumulator = np.zeros((edges.shape[0], edges.shape[1]))
print(accumulator.shape)
# loop over all edge points
for i in range(edge_points.shape[0]):
    for j in range(edge_points.shape[1]):
        if edge_points[i,j] == 255:
            # loop over all possible theta values
            for theta in range(0, 180):
                # convert theta to radians
                theta = theta * np.pi / 180
                # compute rho
                rho = j * np.cos(theta) + i * np.sin(theta)
                # round rho to the nearest integer
                rho = int(round(rho))
                if rho>=540 or rho<=-540: continue
                # increment the accumulator
                accumulator[int(rho), int(theta)] += 1

# find the right lane by finding the peak in hough space
# find the peak in hough space
rho, theta = np.unravel_index(np.argmax(accumulator), accumulator.shape)
xs, ys = create_line(rho, theta, img)
plt.subplot(144)
print(xs, ys)
plt.plot(xs, ys, color='red', linewidth=2)


# create a line from the peak in hough space

# zero out the values in accumulator around the neighborhood of the peak

# find the left lane by finding the peak in hough space


# plot the results
plt.subplot(141),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(mask,cmap = 'gray')
plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(edge_points,cmap = 'gray')
plt.title('Edge Points'), plt.xticks([]), plt.yticks([])
plt.show()