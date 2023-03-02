import numpy as np
import matplotlib.pylab as plt
from skimage import io
from utils import gaussian_kernel, filter2d, partial_x, partial_y
import cv2
import math
def main():
    # Load image
    img = io.imread('iguana.png', as_gray=True)
    ### YOUR CODE HERE
    # Smooth image with Gaussian kernel
    img_smotth = filter2d(img,gaussian_kernel())
    # Compute x and y derivate on smoothed image
    img_deriv_x = partial_x(img_smotth)
    img_deriv_y = partial_y(img_smotth)
    # Compute gradient magnitude
    Hi, Wi = img.shape
    img_grad = np.zeros((Hi, Wi))
    for m in range(Hi):
        for n in range(Wi):
            img_grad[m, n] = math.sqrt(img_deriv_x[m, n]** 2+ img_deriv_y[m, n]** 2)
    # Visualize results
    plt.subplot(1, 3, 1)
    plt.title("gradient image on x direction")
    plt.imshow(img_deriv_x,cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title("gradient image on y direction")
    plt.imshow(img_deriv_y,cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title("gradient magnitude")
    plt.imshow(img_grad,cmap='gray')
    plt.axis('off')
    plt.show()
    # END YOUR CODE
    
if __name__ == "__main__":
    main()

