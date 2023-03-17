from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray
import numpy as np
from scipy.ndimage import correlate
import sklearn.cluster
from scipy.spatial.distance import cdist

def computeHistogram(img_file, F, textons):
    ### YOUR CODE HERE

    # read image
    img = img_as_float(io.imread(img_file))
    img = rgb2gray(img)
    # compute responses
    responses = np.zeros((img.shape[0], img.shape[1], F.shape[2]))
    for i in range(F.shape[2]):
        responses[:,:,i] = correlate(img, F[:,:,i])
    # compute texton map
    texton_map = np.argmin(cdist(responses.reshape(-1, F.shape[2]), textons), axis=1).reshape(img.shape[0], img.shape[1])
    # compute histogram
    h = np.zeros(textons.shape[0])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[texton_map[i,j]] += 1
    h /= np.sum(h)
    return h, texton_map

    ### END YOUR CODE
    
def createTextons(F, file_list, K):
    ### YOUR CODE HERE

    # compute responses
    responses = np.zeros((len(file_list), 150, 150, F.shape[2]))
    for i in range(len(file_list)):
        img = img_as_float(io.imread(file_list[i]))
        img = rgb2gray(img)
        for j in range(F.shape[2]):
            #print(img.shape, F[:,:,j].shape)
            responses[i,:,:,j] = correlate(img, F[:,:,j])
    responses = responses.reshape(-1, F.shape[2])
    # cluster
    kmeans = sklearn.cluster.KMeans(n_clusters=K, random_state=0).fit(responses)
    print(kmeans.cluster_centers_.shape)
    return kmeans.cluster_centers_

    ### END YOUR CODE