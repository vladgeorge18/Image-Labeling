__authors__ = ['1718986','1719379']
__group__ = 'noneyet'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
from skimage.transform import rescale

from utils import rgb2gray


class KNN:
    def __init__(self, train_data, labels, dist_type='euclidean', add_features=False):
        self.add_features = add_features
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.dist_type = dist_type
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.train_data = self.process_images(train_data)
                
    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = self.process_images(test_data)
                
        # Compute distances
        distances = cdist(test_data, self.train_data, self.dist_type)
        
        # Store the k nearest labels
        self.neighbors = np.argsort(distances, axis=1)[:, :k]
        
        self.neighbors = self.labels[self.neighbors]
        

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        predict_labels = []
        
        for neighbor in self.neighbors:
            label = max(neighbor, key=list(neighbor).count)
                        
            predict_labels.append(label)
                        
        return np.array(predict_labels)
        
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
    
    def process_images(self, images):
        """
        Preprocess the images
        """
        if not isinstance(images, np.ndarray):
            images = np.array(images)
            
        if images.dtype != np.float64:
            images = images.astype(np.float64)
        
        avg_color = []
        pixel_variance = []
        max_pixel_val = []
        min_pixel_val = []
        
        if images.ndim == 4:
            # Compute the average RGB color for each image
            if self.add_features:
                for image in images:
                    r_avg = np.mean(image[:, :, 0])
                    g_avg = np.mean(image[:, :, 1])
                    b_avg = np.mean(image[:, :, 2])
                    
                    avg_color.append([r_avg, g_avg, b_avg])
                
                    # pixel_variance.append([np.var(image[:, :, 0]), np.var(image[:, :, 1]), np.var(image[:, :, 2])])
                    
                    max_pixel_val.append([np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2])])
                    min_pixel_val.append([np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2])])
                    
                # Downsample the images acuratete la fel dar dureaza mai putin
                images = np.array([rescale(image, (0.5, 0.5, 1), anti_aliasing=True) for image in images])  # Images are now (30x40)
                
            images = np.array([rgb2gray(image) for image in images])
            
        images = images.reshape(images.shape[0], -1)
        
        # Add new features
        if self.add_features:
            images = np.hstack((images, np.array(avg_color)))
            #images = np.hstack((images, np.array(pixel_variance))) #scade acuratetea
            images = np.hstack((images, np.array(max_pixel_val)))
            images = np.hstack((images, np.array(min_pixel_val)))
            
        return images