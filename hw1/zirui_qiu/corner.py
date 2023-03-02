import numpy as np
from utils import filter2d, partial_x, partial_y, gaussian_kernel
from skimage.feature import peak_local_max
from skimage.io import imread
import matplotlib.pyplot as plt
import cv2


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """


    response = None
    
    ### YOUR CODE HERE
    I_x = partial_x(img)
    I_y = partial_y(img)
    Ixx = filter2d(I_x**2, gaussian_kernel())
    Ixy = filter2d(I_y*I_x, gaussian_kernel())
    Iyy = filter2d(I_y**2, gaussian_kernel())
    # determinant
    detA = Ixx * Iyy - Ixy ** 2
    # trace
    traceA = Ixx + Iyy
    #response
    response = detA - k * traceA ** 2
    ### END YOUR CODE

    return response

def main():
    img = cv2.imread('building.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ### YOUR CODE HERE
    # Compute Harris corner response

    harris_response = harris_corners(img)
    corners = np.zeros(harris_response.shape)
    # Threshold on response


    for rows, response in enumerate(harris_response):
        for cols, r in enumerate(response):
            if r > 0.01* harris_response.max():
                # this is a corner
                corners[rows, cols] = 255
    
    # Perform non-max suppression by finding peak local maximum
    coordinates = peak_local_max(corners, min_distance=18)
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(harris_response,cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('corner response map on each pixel')

    ax[1].imshow(corners,cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('thresholding on the response')

    ax[2].imshow(img,cmap='gray')
    ax[2].autoscale(False)
    ax[2].plot(coordinates[:, 1], coordinates[:, 0], 'X', color='r')
    ax[2].axis('off')
    ax[2].set_title('Peak local max')


    fig.tight_layout()

    plt.show()


    # cv2.imshow('corner response map on each pixel', harris_response)
    # cv2.imshow('thresholding on the response', img_copy_for_corners)
    # print(coordinates)
    # for coord in coordinates:
    #     cv2.drawMarker(img_copy_for_corners,
    #                 position=coord,
    #                 color=(0,0,0),
    #                 thickness=1,
    #                 markerType=cv2.MARKER_TILTED_CROSS,
    #                 line_type=cv2.LINE_8,
    #                 markerSize=5)
    # cv2.imshow('detected corners after non-maximum suppression', img_copy_for_corners)
    # cv2.waitKey(0)
    ### END YOUR CODE
    
if __name__ == "__main__":
    main()
