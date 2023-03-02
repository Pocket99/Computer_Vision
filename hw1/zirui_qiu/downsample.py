import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from utils import gaussian_kernel, filter2d

def main():
    
    # load the image
    im = imread('paint.jpg').astype('float')
    im = im / 255

    # number of levels for downsampling
    N_levels = 5

    # make a copy of the original image
    im_subsample = im.copy()

    # naive subsampling, visualize the results on the 1st row
    for i in range(N_levels):
        #subsample image 
        im_subsample = im_subsample[::2, ::2, :]
        plt.subplot(2, N_levels, i+1)
        plt.imshow(im_subsample)
        plt.axis('off')
        
    # subsampling without aliasing, visualize results on 2nd row
    #### YOUR CODE HERE
    im_subsample_anti = im.copy()
    for i in range(N_levels):
        #subsample image 
        im_subsample_anti= im_subsample_anti[::2, ::2,:]
        plt.subplot(2, N_levels, i+6)
        for j in range(im_subsample_anti.shape[2]):
            im_subsample_anti[:,:,j] = filter2d(im_subsample_anti[:,:,j],gaussian_kernel())
        
        plt.imshow(im_subsample_anti)
        plt.axis('off')
        
    plt.show()
    #### END YOUR CODE
    
if __name__ == "__main__":
    main()
