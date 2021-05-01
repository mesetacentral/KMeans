__authors__ = ['1569031', '1564567', "1571458"]
__group__ = 'Grup03'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    #  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the smaple space is the length of
                    the last dimension
        """

        # check all values are float
        self.X = X 

        if self.X.dtype == np.dtype('uint8'):  # float64
            self.X = self.X.astype('float64')
        else:
            raise ValueError

        # check matrix dimensions are correct
        if len(self.X.shape) == 3:
            self.X = np.reshape(self.X, (self.X.shape[0] * self.X.shape[1], self.X.shape[2])) # X.shape[2] = 3 should be
        elif len(X.shape) == 2:
            pass
        else:
            raise ValueError


    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options == None:
            options = {}
        if not 'km_init' in options:
            options['km_init'] = 'first'
        if not 'verbose' in options:
            options['verbose'] = False
        if not 'tolerance' in options:
            options['tolerance'] = 0
        if not 'max_iter' in options:
            options['max_iter'] = np.inf
        if not 'fitting' in options:
            options['fitting'] = 'WCD'  # within class distance.
        if not 'seed' in options:
            options['seed'] = 0
        if not 'DEC_threshold' in options:
            options['DEC_threshold'] = 20

        # If your methods need any other prameter you can add it to the options dictionary
        self.options = options

        #############################################################
        #  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # https://numpy.org/doc/stable/reference/generated/numpy.isin.html

        self.centroids = np.array([])
        

        if self.options['km_init'].lower() == 'first':
            self.centroids = np.array([self.X[0]])

            i = 0 # iterations
            k = 1 # centroids initialized
            while k < self.K:
                if not any(np.equal(self.centroids, self.X[i]).all(1)):
                    self.centroids = np.append(self.centroids, [self.X[i]], axis=0)
                    k += 1 # new centroid added
                i += 1 # self.X[i] has already been dealt with, lets try self.X[i+1]

            self.old_centroids = np.copy(self.centroids) # has to be copied not assigned, otherwise they get linked somehow

        elif self.options['km_init'].lower() == 'random':
            np.random.seed(self.options['seed']) # seed for rng

            k = 0 # centroids
            while k < self.K:
                i = np.random.randint(0, self.X.shape[0]) # generates random index for self.X
                if not any(np.equal(self.centroids, self.X[i]).all(1)):
                    self.centroids = np.append(self.centroids, [self.X[i]], axis=0)
                    k += 1 # new centroid added

            self.old_centroids = np.copy(self.centroids) # has to be copied not assigned, otherwise they get linked somehow
        else:
            raise ValueError

        # TODO: custom selection of initial centroids
    
    def get_labels(self):
        """
        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        matrix_distances = distance(self.X, self.centroids)
        self.labels = np.argmin(matrix_distances, axis=1)
    

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.old_centroids = np.copy(self.centroids) # has to be copied not assigned, otherwise they get linked

        self.centroids = np.array([np.sum(self.X[self.labels == 0], axis=0) / np.count_nonzero(self.labels == 0)])
        # self.centroids = np.empty(shape=(self.K, self.X.shape[1]))

        for k in range(1, self.K):
            self.centroids = np.append(self.centroids, [self.X.mean(axis = 0, where=np.tile(np.array([self.labels == k]).transpose(), (1, self.X.shape[1])))], axis=0)
            # self.centroids[k] = self.X.mean(axis = 0, where=np.tile(np.array([self.labels == k]).transpose(), (1, self.X.shape[1])))
            # self.centroids[k] = np.sum(self.X[self.labels == k], axis=0) / np.count_nonzero(self.labels == k)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        distance_centroids = np.linalg.norm((self.centroids - self.old_centroids), ord=2, axis=1)
        return not (distance_centroids.any() > self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()

        iterations = 0
        while iterations < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            iterations += 1
            if self.converges():
                break

    def whitinClassDistance(self): # withinClassDistance
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.WCD = 0
        self.WCD = np.sum(np.linalg.norm(self.X - self.centroids[self.labels], ord=2, axis=1)) / self.X.shape[0]
    
        return self.WCD


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        #  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.K = 2
        self.fit()
        last_WCD = self.whitinClassDistance()

        for k in range(3, max_K+1):
            self.K = k
            self.fit()
            WCD = self.whitinClassDistance()

            if 100 - 100 * WCD / last_WCD < self.options['DEC_threshold']:
                self.K = k - 1
                break
            last_WCD = WCD


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
    #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    #  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    matrix_distances = np.empty(shape = (X.shape[0], C.shape[0]))

    for i, x in enumerate(X):
        matrix_distances[i] = np.linalg.norm(x - C, ord=2, axis=1)

    return matrix_distances

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    #  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    probs = utils.get_color_prob(centroids)
    colors = [utils.colors[np.argmax(row)] for row in probs]
    
    """
    colors = []
    for row in probs:
        index = np.argmax(row)
        # index = [np.where(row == np.amax(row))[0]][0]
        colors.append(utils.colors[index])"""

    return colors

def get_colors_and_probs(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color laber folllowing the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroind points)

    Returns:
        lables: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    #  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    #  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors), np.zeros_like(utils.colors, dtype="float")