# import other necessary libaries
from utils import create_line, create_mask
import torch
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# load the input image
img = cv.imread('road.jpg', cv.IMREAD_GRAYSCALE)
print(img.shape)
# run Canny edge detector to find edge points
edges = cv.Canny(img,100,200)


# create a mask for ROI by calling create_mask
mask = create_mask(edges.shape[0], edges.shape[1])

# extract edge points in ROI by multipling edge map with the mask
edge_points = edges * mask

# perform Hough transform 
def hough_line(img):
  # Rho and Theta ranges
  thetas = np.deg2rad(np.arange(-90.0, 90.0))
  width, height = img.shape
  diag_len = np.ceil(np.sqrt(width * width + height * height))   # max_dist
  print("diag",diag_len)
  diag_len = int(diag_len)
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)

  # Hough accumulator array of theta vs rho
  accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

  # Vote in the hough accumulator
  for i in range(len(x_idxs)):
    x = x_idxs[i]
    y = y_idxs[i]

    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho = round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len
      accumulator[rho, t_idx] += 1

  return accumulator, thetas, rhos

accumulator, thetas, rhos = hough_line(edge_points)

# find the right lane by finding the peak in hough space
idx = np.argmax(accumulator)
print("argmax",idx)
rho_b = rhos[int(idx / accumulator.shape[1])]
print("rho",rho_b)
theta_b = thetas[idx % accumulator.shape[1]]
print("theta",theta_b)
print ("rho={0:.2f}, theta={1:.0f}".format(rho_b, np.rad2deg(theta_b)))



# zero out the values in accumulator around the neighborhood of the peak
for i in range(4):
    print("max",accumulator[int(idx / accumulator.shape[1]), idx % accumulator.shape[1]])
    accumulator[int(idx / accumulator.shape[1])-10:int(idx / accumulator.shape[1])+10, idx % accumulator.shape[1]-10:idx % accumulator.shape[1]+10] = 0
    idx = np.argmax(accumulator)

# find the left lane by finding the peak in hough space
print("argmax",idx)
rho_o = rhos[int(idx / accumulator.shape[1])]
print("rho",rho_o)
theta_o = thetas[idx % accumulator.shape[1]]
print("theta",theta_o)
print ("rho={0:.2f}, theta={1:.0f}".format(rho_o, np.rad2deg(theta_o)))
xs_o, ys_o = create_line(rho_o, theta_o, edge_points)
xs_b, ys_b = create_line(rho_b, theta_b, edge_points)

#plot the results
plt.subplot(141),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(mask,cmap = 'gray')
plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(edge_points,cmap = 'gray')
plt.title('Edge Points'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img, cmap='gray')
plt.plot(xs_o, ys_o, color='orange', linewidth=2)
plt.plot(xs_b, ys_b, color='blue', linewidth=2)
plt.title('Final Image'), plt.xticks([]), plt.yticks([])
plt.show()

