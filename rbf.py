# -*- coding: utf-8 -*-
"""
Code modified from the following website:

Zenva - Python Machine Learning, Tutorials on Python Machine Learning, Data Science and Computer Vision
"Using Neural Networks for Regression: Radial Basis Function Networks"
28/10/2017 by Mohit Deshpande

url: https://pythonmachinelearning.pro/using-neural-networks-for-regression-radial-basis-function-networks/
"""

import os
import numpy as np
import pickle
from data import generate_crsval_suffix


param = {'num_centers': 45,
         'augmentation': False}
directories = {'sampled_data': 'E:/UEA/Resource/ICVL/precal/'}


def rbf(x, c, s):
    """Apply a radial basis function (rbf) transform.
    
    Input   x: input vector
            c: assigned center
            s: assigned constant ~ standard deviation
    
    Output  transformed vector
    """
    return np.exp(-1 / (2 * s**2) * dist(x, c, axis=0)**2)


def dist(X, Y, axis):
    """Calculate Euclidean distance.
    
    Input   X, Y: matching data
            axis: the dimension of variables
    
    Output  Euclidean distances of each matching data-point (X, Y)
    """
    return np.sqrt( np.sum( (X - Y)**2, axis=axis ) )


def kmeans(X, k):
    """Performs k-means clustering.
    
    Input   X: input data, DIM_DATA x DIM_Variables
            k: Number of clusters
    
    Output  A kx1 array of final cluster centers
    """
 
    # randomly select initial clusters from input data
    dim_data = X.shape[0]
    clusters_id = np.random.choice(dim_data, size=k)
    clusters = X[clusters_id,...]
    prevClusters = clusters.copy()
    converged = False
 
    while not converged:
        """
        compute distances for each cluster center to each point 
        where (distances[i, j] represents the distance between the ith point and jth cluster)
        """
        distances = np.squeeze(dist(X[:,np.newaxis], clusters[np.newaxis,:], axis=-1))
 
        # find the cluster that's closest to each point
        closestCluster = np.argmin(distances, axis=1)
 
        # update clusters by taking the mean of all of the points assigned to that cluster
        for i in range(k):
            pointsForCluster = X[closestCluster == i]
            if len(pointsForCluster) > 0:
                clusters[i] = np.mean(pointsForCluster, axis=0)
 
        # converge if clusters haven't moved
        converged = np.linalg.norm(clusters - prevClusters) < 1e-6
        prevClusters = clusters.copy()
 
    return clusters


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2):
        self.k = k
        self.centers = ()
        self.stds = ()
        self.feature = ()
        
    def train_centers(self, X):
        self.centers = kmeans(X.T, self.k)
        dMax = max([dist(c1, c2, axis=-1) for c1 in self.centers for c2 in self.centers])
        self.stds = np.repeat(dMax / np.sqrt(2*self.k), self.k)
        
    def transformation(self, X):
        dim_data = X.shape[1]
        
        # training
        A = np.array([rbf(X, self.centers[c,:].reshape(-1,1), self.stds[s]) for c, s in zip(range(self.k), range(self.k))])
        A = np.append(np.ones((1,dim_data)), A, axis=0)
    
        self.feature = A
        
        
def train_rbf_net(out_dir, rgb_data, num_centers):
    """Train RBF net, and save to out_dir.
    
    Input   out_dir: output directory
            rgb_data: input rgb data, DIM_Variables x DIM_Data
            num_centers: number of centers
    """
    # training
    print('-- start training RBF centers --')
    Net = RBFNet(k=num_centers)
    Net.train_centers(rgb_data.T)
    print('-- finished training RBF centers --')
    
    with open(out_dir, 'wb') as handle:
        pickle.dump(Net, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for cmode in [1, 3]:
        train_suffix, _ = generate_crsval_suffix(cmode)
        if param['augmentation']:
            train_suffix = train_suffix + '_aug'
        data_dir = os.path.join(directories['sampled_data'], 'sparse_all_data'+train_suffix+'.pkl')
        rgb_data = pickle.load(open(data_dir, 'rb'))['rgb']
        train_rbf_net('./resources/rbf_icvl'+train_suffix+'.pkl', rgb_data, param['num_centers'])
