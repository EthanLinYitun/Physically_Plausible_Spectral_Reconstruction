# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:32:40 2020

@author: vby18pwu
"""

import os
import utils
import numpy as np
import pickle
from data import *
from evaluation_metrics import *


param = {'num_anchors': 1024,
         'num_neighbors': 8196,
         'num_sampling_points': 30000,
         'random_shuffle': True,
         'augmentation': [2.5, 1]}

directories = {'data': 'E:/UEA/Resource/ICVL/',
               'precal': 'E:/UEA/Resource/ICVL/precal/',
               'sparse_label': 'E:/UEA/Resource/ICVL/sparse/'}

make_sure_dir_exist(directories['precal'])
make_sure_dir_exist(directories['sparse_label'])

resources = {'cmf': load_cie64cmf('./resources/', target_wavelength=np.arange(400, 701, 10)),
             'crsval_name_list': [open('./resources/fn_icvl_group_'+i+'.txt').readlines() for i in ['A','B','C','D']],
             'anchors': [loadmat2array('./resources/icvl_anchors.mat', 'anchor_'+ i) for i in ['A', 'B']]}


def cal_training_data(img_list, crsval_mode):
    print("  Preparing Training Images...")
    gt_data = utils.collect_gt_data(directories['data'], img_list, resources['cmf'], 
                                    num_sampling_points=param['num_sampling_points'], rand=param['random_shuffle'],
                                    augment=param['augmentation'])
    rgb_data_norm = utils.normc(gt_data['rgb'])
    
    if crsval_mode in [1, 2]:
        rgb_anchors_norm = utils.normc(resources['anchors'][0] @ resources['cmf'])
    elif crsval_mode in [3, 4]:
        rgb_anchors_norm = utils.normc(resources['anchors'][1] @ resources['cmf'])
    
    nearest_neighbors = utils.knn(rgb_data_norm, rgb_anchors_norm, k=param['num_neighbors'], batch_size=250)
    
    train_suffix, _ = generate_crsval_suffix(crsval_mode)
    #if param['random_shuffle']:
    #    train_suffix = train_suffix + '_rand'
    if param['augmentation']:
        train_suffix = train_suffix + '_aug'
    with open(os.path.join(directories['precal'], 'sparse_all_data'+train_suffix+'.pkl'), 'wb') as handle:
        pickle.dump(gt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    np.save(os.path.join(directories['precal'], 'sparse_neighbor_idx'+train_suffix+'.npy'), nearest_neighbors)
    

def cal_validation_data(img_list, crsval_mode):
    print("  Preparing Validation Images...")
    gt_data = utils.collect_gt_data(directories['data'], img_list, resources['cmf'], 
                                    num_sampling_points=param['num_sampling_points'], rand=param['random_shuffle'])
    rgb_data_norm = utils.normc(gt_data['rgb'])

    if crsval_mode in [1, 2]:
        rgb_anchors_norm = utils.normc(resources['anchors'][0] @ resources['cmf'])
    elif crsval_mode in [3, 4]:
        rgb_anchors_norm = utils.normc(resources['anchors'][1] @ resources['cmf'])
    
    nearest_neighbors = utils.knn(rgb_data_norm, rgb_anchors_norm, k=param['num_neighbors']//2, batch_size=250)
    
    _, val_suffix = generate_crsval_suffix(crsval_mode)
    #if param['random_shuffle']:
    #    val_suffix = val_suffix + '_rand'
    if param['augmentation']:
        val_suffix = val_suffix + '_aug'
    with open(os.path.join(directories['precal'], 'sparse_all_data'+val_suffix+'.pkl'), 'wb') as handle:
        pickle.dump(gt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    np.save(os.path.join(directories['precal'], 'sparse_neighbor_idx'+val_suffix+'.npy'), nearest_neighbors)


def cal_testing_image_labels(img_list, crsval_mode):
    print("  Preparing Testing Images...")
    if crsval_mode in [1, 2]:
        rgb_anchors_norm = utils.normc(resources['anchors'][0] @ resources['cmf'])
    elif crsval_mode in [3, 4]:
        rgb_anchors_norm = utils.normc(resources['anchors'][1] @ resources['cmf'])
        
    for img_name in img_list:
        print("    ", img_name[:-1])
        spec_img = load_icvl_data(directories['data'], img_name[:-1]) # 31 x H x W
        gt_data = {} 
        gt_data['spec'], gt_data['rgb'] = utils.cal_gt_data(spec_img, resources['cmf'])
        
        rgb_data_norm = utils.normc(gt_data['rgb'])
        nearest_anchor = utils.knn(rgb_anchors_norm, rgb_data_norm, k=1, batch_size=400000).reshape(-1)
        
        np.save(os.path.join(directories['sparse_label'], img_name[:-5]+'_label.npy'), nearest_anchor)


if __name__ == '__main__':    
    print("Start preparing sparse coding...")
    for cmode in [1, 2, 3, 4]:
        print("Cross Validation", cmode)
        train_list, val_list, test_list = generate_crsval_imlist(resources['crsval_name_list'], crsval_mode=cmode)
        if cmode in [1, 3]:
            cal_training_data(train_list, cmode)
        cal_validation_data(val_list, cmode)
        cal_testing_image_labels(test_list, cmode)
    