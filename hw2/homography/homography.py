import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage import data
from skimage import transform
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
    ### cv method
    # sift = cv.SIFT_create()
    # kp1, des1 = sift.detectAndCompute(I1, None)
    # kp2, des2 = sift.detectAndCompute(I2, None)
    # locs1 = np.array([kp1[i].pt for i in range(len(kp1))])
    # locs2 = np.array([kp2[i].pt for i in range(len(kp2))])
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des1, des2, k=2)
    # good = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         good.append(m)


    # matches = np.array([[m.queryIdx, m.trainIdx] for m in good])
    ### END YOUR CODE
    
    return matches, locs1, locs2

def computeH_ransac(matches, locs1, locs2):

    # Compute the best fitting homography using RANSAC given a list of matching pairs
    ### YOUR CODE HERE
    ### You should implement this function using Numpy only
    # The raw matching result contains outliers. This function
    # implements the RANSAC algorithm for estimating the homography. You are required to implement this
    # function from scratch using only Numpy (e.g. do not use any additional function from skimage, OpenCV,
    # or other libraries). You might find the numpy.linalg to be useful, e.g. for doing SVD decomposition. This
    # function has 2 outputs: bestH, inliers as follows:
    # bestH: this is a 3*3 matrix representing the estimated homography matrix
    # inliers: this is an array indicating which candidate match is an inlier. For example, if there are 20
    # inliers, inliers will be an array of length 20. If inliers[0]=5, it means the first inlier is given by matches[5,:]
    # (matches[5,0] is the index of the match in locs1, and matches[5,1] is the index of the match in locs2)
    # Compute the best fitting homography given a list of matching points
    max_iters = 1000 # the number of iterations to run RANSAC for
    inlier_tol = 0.5# the tolerance value for considering a point to be an inlier
    bestH = np.empty([3, 3])
    rand_1 = np.empty ([2, 4])
    rand_2 = np.empty ([2, 4])
    max_inliers = -1
    # only consider candidate matches
    # NOTE: Locs1 store [row, col], but here we need [x,y]
    x1 = locs1[matches [:, 0], ::-1]
    x2 = locs2[matches [:, 1], ::-1]
    x1_hom = np.hstack((x1, np.ones ((x1.shape[0], 1))))
    x2_hom = np.hstack((x2, np.ones ((x2.shape[0], 1))))
    for iter in range (max_iters):
        ind_rand = np. random. choice (x1.shape[0], 4, replace=False)
        rand_1 = x1[ind_rand, :]
        rand_2 = x2[ind_rand, :]

        A = np. empty ([2*rand_1. shape[0], 9])
        for ind in range (rand_1.shape[0]):
            u_1 = rand_1[ind, 0]
            v_1 = rand_1[ind, 1]
            u_2 = rand_2[ind, 0]
            v_2 = rand_2[ind, 1]
            A[2*ind] = [-u_1,-v_1,-1,0,0,0,u_1*u_2,v_1*u_2,u_2]
            A[2*ind+1] = [0,0,0,-u_1,-v_1,-1,u_1*v_2,v_1*v_2,v_2]
        U, S, V_t=np.linalg.svd(A)
        eig_vec = V_t[-1, :]
        eig_vec = V_t[-1, :] / V_t[-1, -1]
        H = eig_vec.reshape(3, 3)

        pred_x2 = np.matmul (x1_hom, H.T)
        pred_x2 = pred_x2 / pred_x2[:, 2:3]
        error = pred_x2 - x2_hom
        error = np.linalg.norm(error, axis=1)
        curr_inliers = np.where(error < inlier_tol)[0]
        tot_inlier = curr_inliers.size
        if tot_inlier > max_inliers:
            max_inliers = tot_inlier
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
    # N = 1000
    # tol = 2
    # bestH = np.zeros((3, 3))
    # inliers = []
    # for i in range(N):
    #     # randomly select 4 points
    #     idx = np.random.choice(matches.shape[0], 4, replace=False)
    #     # get the corresponding points
    #     p1 = locs1[matches[idx, 0], :]
    #     p2 = locs2[matches[idx, 1], :]
    #     # compute the homography matrix
    #     A = np.zeros((8, 9))
    #     for j in range(4):
    #         A[2 * j, :] = [p1[j, 0], p1[j, 1], 1, 0, 0, 0, -p2[j, 0] * p1[j, 0], -p2[j, 0] * p1[j, 1], -p2[j, 0]]
    #         A[2 * j + 1, :] = [0, 0, 0, p1[j, 0], p1[j, 1], 1, -p2[j, 1] * p1[j, 0], -p2[j, 1] * p1[j, 1], -p2[j, 1]]
    #     U, S, V = np.linalg.svd(A)
    #     H = V[-1, :].reshape(3, 3)
    #     # compute the inliers
    #     p1 = np.concatenate((locs1[matches[:, 0], :], np.ones((matches.shape[0], 1))), axis=1)
    #     p2 = np.concatenate((locs2[matches[:, 1], :], np.ones((matches.shape[0], 1))), axis=1)
    #     p2_hat = np.dot(H, p1.T).T
    #     p2_hat = p2_hat / p2_hat[:, 2].reshape(-1, 1)
    #     dist = np.sqrt(np.sum((p2 - p2_hat) ** 2, axis=1))
    #     idx = np.where(dist < tol)[0]
    #     if len(idx) > len(inliers):
    #         inliers = idx
    #         bestH = H