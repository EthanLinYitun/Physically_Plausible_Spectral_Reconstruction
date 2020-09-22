# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 12:19:10 2020

@author: vby18pwu
"""

import poly
import os
import numpy as np
from numpy.linalg import inv, det
import pickle
import matplotlib.pyplot as plt
from data import generate_file_name, load_icvl_data
from scipy.spatial.distance import cdist
import hscnn
#import utils_h as hscnn
  
def cal_gt_data(spectral_img, cmf, num_sampling_points=(), rand=False, augment=False):
    spec_data = spectral_img.reshape(spectral_img.shape[0], -1).T # Dim_Data x 31    
    if num_sampling_points:
        spec_data = sampling_data(spec_data, num_sampling_points, rand)
    else:
        num_sampling_points = spec_data.shape[0]
    
    if augment:
        all_spec_data = spec_data * gen_random_scale(base=augment[0], img_shape=(num_sampling_points, 1))
    else:
        all_spec_data = spec_data
    
    rgb_data = all_spec_data @ cmf
    
    return all_spec_data, rgb_data


def collect_gt_data(dir_data, img_list, cmf, num_sampling_points, rand=True, augment=False):
    gt_data = {'spec': [],
               'rgb': []}
    
    for img_name in img_list:
        spec_img = load_icvl_data(dir_data, img_name[:-1]) # 31 x H x W
        spec_data, rgb_data = cal_gt_data(spec_img, cmf, num_sampling_points, rand, augment)
        
        gt_data['spec'] = gt_data['spec'] + [spec_data]
        gt_data['rgb']  = gt_data['rgb']  + [rgb_data]
    
    gt_data['spec'] = np.array(gt_data['spec']).swapaxes(0, 2).reshape(spec_data.shape[1], -1).T
    gt_data['rgb']  = np.array(gt_data['rgb']).swapaxes(0, 2).reshape(rgb_data.shape[1], -1).T
    
    return gt_data


def apply_random_scale(spec_data, base):
    dim_data = spec_data.shape[0]
    return spec_data * gen_random_scale(base=base, img_shape=(dim_data, 1))


def normc(data):
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def knn(data, reference, k, batch_size=None):
    num_reference = reference.shape[0]
    if batch_size:
        pass
    else:
        batch_size = num_reference
    num_batch = num_reference//batch_size
    num_residual = num_reference%batch_size
    out_list = np.zeros((num_reference, k))
    for i in range(num_batch):
        D = cdist(reference[i*batch_size:(i+1)*batch_size, :], data)
        out_list[i*batch_size:(i+1)*batch_size, :] = np.argsort(D, axis=1)[:, :k]
    if num_residual:
        D = cdist(reference[-num_residual:, :], data)
        out_list[-num_residual:, :] = np.argsort(D, axis=1)[:, :k]
    
    return out_list
    

def nearest_neighbor(data, anchors, k):
    
    D = cdist(data, anchors)
    
    return np.argmin(D, axis=1)
    
    
    num_data = data.shape[0]
    idx_list = []
    for i in range(num_data):
        dist = np.sum((anchors - data)**2, axis=1, keepdims=False)
        idx_list.append(np.argmin(dist))
    
    return idx_list

def rgb2poly(rgb_data, poly_order, root):
    
    dim_data, dim_variables = rgb_data.shape
    poly_term = poly.get_polynomial_terms(dim_variables, poly_order, root)
    dim_poly = len(poly_term)
    
    out_mat = np.empty((dim_data, dim_poly))
    
    for term in range(dim_poly):
        new_col = np.ones((dim_data))            # DIM_DATA,
        for var in range(dim_variables):
            variable_vector = rgb_data[:, var]                             # DIM_DATA,
            variable_index_value = poly_term[term][var]
            new_col = new_col * ( variable_vector**variable_index_value )
            
        out_mat[:,term] = new_col
    
    return out_mat


def rgb2rbf(rgb_data, rbf_net):
    rbf_net.transformation(rgb_data.T)
    rfb_data = rbf_net.feature.T
    
    return rfb_data


def spec2null(spec_data, null_basis):
    return ( inv(null_basis.T @ null_basis) @ null_basis.T @ spec_data.T ).T


def data_transform(gt_data, regress_mode, advanced_mode, crsval_mode, resources=(), exposure=1):
    # Transformation on RGB data
    if regress_mode['type'] == 'poly':
        regress_input = rgb2poly(gt_data['rgb']*exposure, regress_mode['order'], root=False)
    elif regress_mode['type'] == 'root-poly':
        regress_input = rgb2poly(gt_data['rgb']*exposure, regress_mode['order'], root=True)
    elif regress_mode['type'] == 'rbf':
        if crsval_mode in [1, 2]:
            if advanced_mode['Data_Augmentation']:
                regress_input = rgb2rbf(gt_data['rgb']*exposure, resources['rbf_net'][2])
            else:
                regress_input = rgb2rbf(gt_data['rgb']*exposure, resources['rbf_net'][0])
        elif crsval_mode in [3, 4]:
            if advanced_mode['Data_Augmentation']:
                regress_input = rgb2rbf(gt_data['rgb']*exposure, resources['rbf_net'][3])
            else:
                regress_input = rgb2rbf(gt_data['rgb']*exposure, resources['rbf_net'][1])
    
    # Transformation on Spectral data
    if advanced_mode['Physically_Plausible']:        
        regress_output = spec2null(gt_data['spec']*exposure, resources['null_basis'])
    else:
        regress_output = gt_data['spec']*exposure
    
    return regress_input, regress_output


def recover(regress_matrix, regress_input, advanced_mode, resources, gt_rgb=(), exposure=1):
    
    recovery = {}
    recovery['spec'] = regress_input @ regress_matrix
    if advanced_mode['Physically_Plausible']:
        recovery['spec'] = gt_rgb*exposure @ resources['funda_mat'].T + recovery['spec'] @ resources['null_basis'].T
    
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
    
    return recovery

def get_regression_parts(data_spec, data_from_rgb, weights=()):
    '''
    Input data_spec with shape ( DIM_DATA, DIM_SPEC )
          data_from_rgb with shape ( DIM_DATA, -1 ), could be data_poly or data_patch
    Output squared_term, body_term
    '''
    
    if weights == ():
        squared_term = data_from_rgb.T @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = data_from_rgb.T @ data_spec      # DIM_RGB x DIM_SPEC
    else:
        weights = np.diag(weights)
        
        squared_term = data_from_rgb.T @ weights @ data_from_rgb    # DIM_RGB x DIM_RGB
        body_term = data_from_rgb.T @ weights @ data_spec      # DIM_RGB x DIM_SPEC
    
    return squared_term, body_term


class RegressionMatrix():
    def __init__(self, regress_mode, advanced_mode):
        
        self.regress_mode = regress_mode
        self.advanced_mode = advanced_mode
        
        if regress_mode['type'] == 'rbf':
            self.__dim_regress_input = regress_mode['num_centers'] + 1
        elif regress_mode['type'] == 'poly':
            self.__dim_regress_input = len(poly.get_polynomial_terms(regress_mode['dim_rgb'], regress_mode['order'], False))
        elif regress_mode['type'] == 'root-poly':
            self.__dim_regress_input = len(poly.get_polynomial_terms(regress_mode['dim_rgb'], regress_mode['order'], True))
        
        if advanced_mode['Physically_Plausible']:
            self.__dim_regress_output = regress_mode['dim_spec'] - regress_mode['dim_rgb']
        else:
            self.__dim_regress_output = regress_mode['dim_spec']
            
        self.__squared_term = np.zeros((self.__dim_regress_input, self.__dim_regress_input))
        self.__body_term = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
        self.__matrix = np.zeros((self.__dim_regress_input, self.__dim_regress_output))
        self.__gamma = 1
            
    def set_gamma(self, gamma):
        self.__gamma = gamma
        self.__matrix = inv( self.__squared_term + self.__gamma * np.eye(self.__dim_regress_input) ) @ self.__body_term
    
    def get_gamma(self):
        return self.__gamma
    
    def get_matrix(self):
        return self.__matrix
    
    def get_dim_regress_input(self):
        return self.__dim_regress_input
    
    def get_dim_regress_output(self):
        return self.__dim_regress_output
    
    def test_feasible_gamma(self, gamma):
        return det(self.__squared_term + gamma * np.eye(self.__dim_regress_input)) != 0
    
    def update(self, regress_input, regress_output):
        squared_term, body_term = get_regression_parts(regress_output, regress_input)
        self.__squared_term = self.__squared_term + squared_term
        self.__body_term = self.__body_term + body_term
            
            
def save_model(RegMat, dir_model, regress_mode, advanced_mode, model_suffix):
    fn = generate_file_name(regress_mode, advanced_mode)
    with open(os.path.join(dir_model, fn + model_suffix + '.pkl'), 'wb') as handle:
        pickle.dump(RegMat, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_model(dir_model, regress_mode, advanced_mode, model_suffix):
    fn = generate_file_name(regress_mode, advanced_mode)
    with open(os.path.join(dir_model, fn + model_suffix + '.pkl'), 'rb') as handle:
        return pickle.load(handle)

def sampling_data(data, num_sampling_points, rand=False):
    if rand:
        np.random.shuffle(data)
        
    sampling_points = np.floor(np.linspace(0, len(data), num_sampling_points, endpoint=False)).astype(int)
    return data[sampling_points, :]


def regularize(RegMat, regress_input, gt_data, advanced_mode, cost_func, resources=(), show_graph=False):
    
    def determine_feasible_gamma(channel=()):
        for s in range(-20, 0, 1):
            if RegMat.test_feasible_gamma(10**s):
                break
        return np.logspace(-s, s, 20)
    
    def regularizer(test_gammas):
        cost = []
        for gamma in test_gammas:
            RegMat.set_gamma(gamma)        
            recovery = recover(RegMat.get_matrix(), regress_input, advanced_mode, resources, gt_data['rgb'])  
            cost.append(np.mean(cost_func(gt_data, recovery)))
            
        best_gamma = test_gammas[np.argmin(cost)]
        
        if show_graph:
            plt.figure()
            plt.title('Tikhonov parameter search')
            plt.plot(test_gammas, cost)
            plt.scatter(best_gamma, np.min(cost), c='r', marker='o')
            plt.xscale('log')
            plt.show()
        
        return best_gamma
    
    test_gammas = determine_feasible_gamma()
    best_gamma = regularizer(test_gammas)
    test_gammas_fine = best_gamma * np.logspace(-1, 1, 1000)
    best_gamma = regularizer(test_gammas_fine)
    RegMat.set_gamma(best_gamma)
    
    return RegMat


def gen_random_scale(base=2.5, rnd=0, img_shape=()):
    
    if rnd != 0:
        np.random.seed(rnd)
    
    if len(img_shape):
        assert len(img_shape) == 2, 'Input image shape should be 2-D'
        log_scale = (np.random.rand(img_shape[0],img_shape[1])-0.5)*2  

    else:
        log_scale = (np.random.rand(1)[0]-0.5)*2    
    
    return base**log_scale


def recover_HSCNN_R(rgb, img_shape, resources, model, model_type, exposure):
    
    dim_spec, height, width = img_shape

    rgb = np.array(rgb).T.reshape(3, height, width) 	# 3 x height x width
    rgb = np.swapaxes(np.swapaxes(rgb, 0, 1), 1, 2 )		# height x 3 x width
    
    curr_rgb = (rgb * exposure).astype('float32')
    curr_rgb = np.expand_dims(np.transpose(curr_rgb,[2,1,0]), axis=0).copy() 
    
    img_res1 = hscnn.reconstruction(curr_rgb, model)
    img_res2 = np.flip(hscnn.reconstruction(np.flip(curr_rgb, 2).copy(), model), 1) 
   
    img_res3 = (img_res1+img_res2)/2
    
    if model_type in ['orig','dataug']:
        final_img = np.swapaxes(np.swapaxes(img_res3/4095,0,2),1,2)
        
    elif model_type == 'color':
        final_img = resources['funda_mat'] @ curr_rgb.reshape(3,-1) + resources['null_basis'] @ ((np.swapaxes(img_res3,0,2).reshape(28,-1)/4095)*2-1)
        final_img = np.swapaxes(np.array(final_img).reshape(31,width,height),1,2) # 31 x height x width

    elif model_type == 'color_dataug':
        final_img = resources['funda_mat'] @ curr_rgb.reshape(3,-1) + resources['null_basis'] @ ((np.swapaxes(img_res3,0,2).reshape(28,-1)/4095)*20-10)
        final_img = np.swapaxes(np.array(final_img).reshape(31,width,height),1,2) # 31 x height x width
        
    recovery = {}
    recovery['spec'] = final_img.reshape(dim_spec,-1).T
    recovery['rgb'] = recovery['spec'] @ resources['cmf']
        
    return recovery
    
    
    