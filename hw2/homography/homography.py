import numpy as np
import cv2 as cv
from skimage.feature import match_descriptors, SIFT
from skimage.color import rgb2gray
def matchPics(I1, I2):
    # Given two images I1 and I2, perform SIFT matching to find candidate match pairs
    ### YOUR CODE HERE
    ### You can use skimage or OpenCV to perform SIFT matching
    I2 = rgb2gray(I2)
    sift = SIFT()
    sift.detect_and_extract(I1)
    kp1, des1 = sift.keypoints, sift.descriptors
    sift.detect_and_extract(I2)
    kp2, des2 = sift.keypoints, sift.descriptors
    matches = match_descriptors(des1, des2, max_ratio=0.5, cross_check=True)
    locs1 = np.array([kp1[i] for i in range(len(kp1))])
    locs2 = np.array([kp2[i] for i in range(len(kp2))])
    ### END YOUR CODE
    
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    # Compute the best fitting homography using RANSAC given a list of matching pairs
    ### YOUR CODE HERE
    N = 1000 # the number of iterations
    inlier_tol = 0.5 # tolerance value for points to be considered as inliers
    bestH = np.zeros([3, 3]) 
    inliers = []
    # get matched keypoints
    k1 = locs1[matches[:, 0], ::-1]
    k2 = locs2[matches[:, 1], ::-1]

    for iter in range (N):
        # Randomly select 4 matches
        idx_rand = np.random.choice(k1.shape[0], 4, replace=False)
        # get the corresponding points
        k1_rand = k1[idx_rand, :]
        k2_rand = k2[idx_rand, :]
        # Compute the homography
        A = np.zeros([8, 9])
        for j in range (4):
            # fill in the A matrix
            A[2*j, :] = [-k1_rand[j, 0], -k1_rand[j, 1], -1, 0, 0, 0, k1_rand[j, 0]*k2_rand[j, 0], k1_rand[j, 1]*k2_rand[j, 0], k2_rand[j, 0]] 
            A[2*j+1, :] = [0, 0, 0, -k1_rand[j, 0], -k1_rand[j, 1], -1, k1_rand[j, 0]*k2_rand[j, 1], k1_rand[j, 1]*k2_rand[j, 1], k2_rand[j, 1]]
        U, S, V=np.linalg.svd(A)
        H = V[-1, :].reshape(3, 3) # Store singular vector of the smallest singular value, Reshape to get H
        # add 1 to the end of each row of p1 and p2 to convert to homogeneous coordinates
        p1_hom = np.hstack((k1, np.ones((k1.shape[0], 1)))) 
        p2_hom = np.hstack((k2, np.ones((k2.shape[0], 1)))) 
        # p_hat = H * p1 Convert p’ from homogeneous to image coordinates
        p_hat = np.matmul(p1_hom, H.T) 
        # Compute inliers
        p_hat = p_hat / p_hat[:, 2].reshape(-1,1) # normalize
        dist = np.linalg.norm(p_hat - p2_hom, axis=1) # compute the distance between p_hat and p2
        curr_inliers = np.where(dist < inlier_tol)[0] # find the inliers
        # Update the number of inliers and the best fitting homography
        if len(curr_inliers) > len(inliers):
            bestH = H
            inliers = curr_inliers
                    
    

    ### END YOUR CODE

    return bestH, inliers

def compositeH(H, template, img):

    # Create a compositie image after warping the template image on top
    # of the image using homography

    #Create mask of same size as template
    mask = np.ones(template.shape)

    #Warp mask by appropriate homography
    warped_mask = cv.warpPerspective(mask, H, (img.shape[1], img.shape[0]))

    #Warp template by appropriate homography
    warped_template = cv.warpPerspective(template, H, (img.shape[1], img.shape[0]))

    #Use mask to combine the warped template and the image
    composite_img = warped_template * warped_mask + img * (1 - warped_mask)
    
    return composite_img
