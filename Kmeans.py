__authors__ = ['1718986','1719379']
__group__ = 'noneyet'

import numpy as np
import utils
import matplotlib.pyplot as plt
import random
class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 50
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.labels = []
    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################
        

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        
        if X.ndim == 3:
            X = X.reshape(-1,X.shape[2])
            
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #np.random.seed(42)
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids = np.zeros((self.K,self.X.shape[1]))
        
        if self.options['km_init'].lower() == 'first':
            ct = 0
            is_centroid = []
            for pixel in self.X:
                if not any((pixel == centroid).all() for centroid in is_centroid):
                    is_centroid.append(pixel)
                    self.old_centroids[ct] = self.centroids[ct] = pixel 
                    ct +=1

                if ct == self.K:
                    break
        elif self.options['km_init'].lower() == 'random':
            #print(self.K)
            unique_points = np.unique(self.X, axis = 0)
            idx = np.random.choice(unique_points.shape[0], self.K, replace=False)
            self.old_centroids = self.centroids = unique_points[idx]
            #print(self.centroids)
            
        elif self.options['km_init'].lower() == 'custom':
            # Implement K-means++ initialization
            self.old_centroids = self.centroids = np.zeros(shape=(self.K,3))
            idx = np.random.choice(len(self.X))
            self.old_centroids[0] = self.centroids[0]=self.X[idx]
            for i in range(1,self.K):
                distances = np.array([np.min(np.linalg.norm(self.centroids[:i] - pixel,axis = 1)) for pixel in self.X])
                probabilites = distances / np.sum(distances)
                idx = np.random.choice(self.X.shape[0],p = probabilites)
                #idx = np.argmax(distances)
                self.old_centroids[i] = self.centroids[i] = self.X[idx]
        
        elif self.options['km_init'].lower() == 'custom1':
            #Implement hypercube initialization
            #hypercube: k1 x k2 x .. x kd, where d is the number of features
            # and every ki is splitted in k subsets
            # a centroid is sampled by the following rule:
            # from every ki we choose a subset from where we consider the mean as being the ith point of the centroid
            # every combination of subsets is unique

            #compute a set for every feature
            feature_sets = np.zeros((self.X.shape[1],self.X.shape[0]))
            for i in range(self.X.shape[1]):
                feature_sets[i] = self.X[:,i]
            
            #compute partitions splitting every feature set in k subsets
            partitions = []
            for i in range(self.X.shape[1]):
                partition = np.array_split(feature_sets[i],self.K)
                partitions.append(partition)
            
            #compute the tuples for choosing the subset for every partition
            tuples = []
            while len(tuples) < self.K:
                new_tuple = tuple(np.random.randint(0, self.K, size=self.X.shape[1]))
                if new_tuple not in tuples:
                    tuples.append(new_tuple)

            #compute the centroids
            centroids = []
            for tup in tuples:
                centroid = []
                for idx,elem in enumerate(tup):
                    #centroid.append(random.choice(partitions[idx][elem]))
                    centroid.append(np.mean(partitions[idx][elem]))
                centroids.append(centroid)

            self.old_centroids = self.centroids = np.array(centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.labels = np.random.randint(self.K, size=self.X.shape[0])
        #self.labels = np.array([np.argmin(np.linalg.norm(self.centroids-pixel,axis=1)) for pixel in self.X])
        distances = np.linalg.norm(self.X[:, None] - self.centroids, axis=2)
        self.labels = np.argmin(distances, axis=1)
        #print(set(self.labels))
        while len(set(self.labels)) != len(self.centroids):
            label_set = set(self.labels)
            for i in range(self.K):
                if i not in label_set:
                    elem = random.choice(self.X)
                    while elem in self.centroids:
                        elem = random.choice(self.X)
                    self.centroids[i] = elem
            
            distances = np.linalg.norm(self.X[:, None] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)
        
        #print(set(self.labels))
            
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.old_centroids = self.centroids.copy()
        self.centroids = np.array([np.mean(self.X[self.labels == k], axis=0) for k in range(self.K)])

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if np.sum(np.abs(self.centroids-self.old_centroids)) < 0.01:
            return True
        else:
            return False

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #print(self.K)
        #print("ok0")
        self._init_centroids()
        #print("ok1")
        for i in range(self.num_iter):
            self.get_labels()
            
            self.get_centroids()
           # print("ok2")
            if self.converges():
                break

            self.iter = i + 1
        

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distances = np.square(np.linalg.norm(self.X - self.centroids[self.labels],axis=1))
        WCD = np.sum(distances) / len(self.X)
        return WCD

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        wcds = []
        for k in range(0,max_K-1):
            self.K = k+1
            self.fit()
            wcds.append(self.withinClassDistance())
        results = [100 - 100*(wcds[k]/wcds[k-1])+1 for k in range(1,max_K-1)]
        #print(results)
        for i in range(len(results)):
            if results[i] < 20:
                self.K = i+ 1
                #print(self.K)
                break
    
    def get_precentage(self):
        idx = set(self.labels)
        prc = np.zeros(len(idx))
        for elem in self.labels:
            prc[elem]+=1

        return prc / (len(self.X))

        
         
def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dist = np.sqrt(np.sum(np.square((X[:, None] - C)),axis=2))
    return dist

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    prob = utils.get_color_prob(centroids)
    
    idx = np.argmax(prob,axis=1)
    
    #print(idx)
    return list(utils.colors[idx])



    
