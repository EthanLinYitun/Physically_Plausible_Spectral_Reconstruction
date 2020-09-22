# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 16:38:23 2020

@author: vby18pwu
"""

import h5py
import numpy as np
from matplotlib.pyplot import imread
import os
from pandas import read_csv
import csv
import datetime
import pickle
from glob import glob
import torch
import torch.nn as nn
import hscnn

def loadmat2array(directory, name=[]):
    
    f = h5py.File(directory,'r')
    if name == []:
        for kname, obj in f.items():
            print(kname)
        name = input('Type in key names: ')
    
    for kname, obj in f.items():
        if kname == name:
            output_obj = obj
            break
    
    return np.array(output_obj)


def load_icvl_data(directory, img_name):
    return loadmat2array(os.path.join(directory, img_name), 'rad') / 4095 # 31 x 1392 x 1300


def load_cave_data(directory, img_name):
    out_img = np.zeros((31,512,512))
    for i in range(31):
        ch_img = imread(os.path.join(directory, img_name, img_name, img_name+'_{0:02d}.png'.format(i+1)))
        if ch_img.ndim != 2:
            ch_img = ch_img[:,:,0]
        out_img[i,:,:] = ch_img
    
    return out_img  #range: [0,1]


def load_cie64cmf(directory, target_wavelength=np.arange(400,701,10)):
    path_name = os.path.join(directory, 'ciexyz64.csv')

    cmf = np.array(read_csv(path_name))[:,1:]
    lambda_cmf = np.array(read_csv(path_name))[:,0]   # (365:830:5)
    
    cmf = interpolate(cmf, lambda_cmf, target_wavelength)
    cmf = cmf / np.max(np.sum(cmf, 0))
    
    return cmf


def load_camera_csf(directory, camera_name, target_wavelength=np.arange(400,701,10)):
    all_available_camera = {'sony_imx135', 'nikon_d810', 'canon_5d'}
    
    if camera_name == 'sony_imx135':
        with open(os.path.join(directory, 'csf_sonyimx135.pkl'), 'rb') as handle:
            d = pickle.load(handle)
        csf = interpolate(d['csf'], d['wl'], target_wavelength)
        csf = csf / np.max(np.sum(csf, 0))
    
    elif camera_name == 'nikon_d810':
        with open(os.path.join(directory, 'csf_nikond810.pkl'), 'rb') as handle:
            d = pickle.load(handle)
        csf = interpolate(d['csf'], d['wl'], target_wavelength)
        csf = csf / np.max(np.sum(csf, 0))
    
    elif camera_name == 'canon_5d':
        with open(os.path.join(directory, 'csf_canon5d.pkl'), 'rb') as handle:
            d = pickle.load(handle)
        csf = interpolate(d['csf'], d['wl'], target_wavelength)
        csf = csf / np.max(np.sum(csf, 0))
    
    else:
        print('No illuminant name', camera_name)
        print('Please type in one of the following name', all_available_camera)
        camera_name = input()
        
        load_camera_csf(directory, camera_name, target_wavelength)
        
    return csf


def load_illuminant(directory, illuminant_name, target_wavelength=np.arange(400,701,10)):
    
    all_available_illum = {'A', 'D65', 'E'}
    
    if illuminant_name == 'A':
        illum = np.array(read_csv(os.path.join(directory, 'Illuminant_A.csv')))[:,1]
        lambda_illum = np.array(read_csv(os.path.join(directory, 'Illuminant_A.csv')))[:,0] # (300:780:5)
        
    elif illuminant_name == 'D65':
        illum = np.array(read_csv(os.path.join(directory, 'Illuminant_D65.csv')))[:,1]
        lambda_illum = np.array(read_csv(os.path.join(directory, 'Illuminant_D65.csv')))[:,0] # (300:830:5)
        
    elif illuminant_name == 'E':        
        illum = np.ones(target_wavelength.size)     # E illuminant
        lambda_illum = target_wavelength
    
    else:
        print('No illuminant name', illuminant_name)
        print('Please type in one of the following name', all_available_illum)
        illuminant_name = input()
        
        load_illuminant(directory, illuminant_name, target_wavelength)
    
    illum = interpolate(illum, lambda_illum, target_wavelength)
    illum = illum / np.max(illum)
    
    return illum

def interpolate(data, data_waveL, targeted_waveL):
    
    assert data.shape[0] == data_waveL.size, 'Wavelength sequence mismatch with data'
    
    targeted_bounds = [np.min(targeted_waveL), np.max(targeted_waveL)]
    data_bounds = [np.min(data_waveL), np.max(data_waveL)]
    
    assert data_bounds[0] <= targeted_bounds[0], 'targeted wavelength range must be within the original wavelength range'
    assert data_bounds[1] >= targeted_bounds[1], 'targeted wavelength range must be within the original wavelength range'
    
    dim_new_data = list(data.shape)
    dim_new_data[0] = len(targeted_waveL)
    new_data = np.empty(dim_new_data)
    
    for i in range(len(targeted_waveL)):

        relative_L = data_waveL - targeted_waveL[i]
        
        if 0 in relative_L:
            floor = np.argmax( relative_L == 0 )
            new_data[i,...] = data[floor,...]
        
        else:
            floor = np.argmax( relative_L >= 0 ) -1
            interval = data_waveL[floor+1] - data_waveL[floor]
            portion = (targeted_waveL[i] - data_waveL[floor])/interval
            new_data[i,...] = portion*data[floor,...] + (1-portion)*data[floor+1,...]
    
    return new_data 


def load_icvl_white_point(wp_list, name_list, img_name):    
    for j in range(len(name_list)):
        if name_list[j] == img_name:
            selected_j = j
            break
    else:
        assert False, "Image's name not on the white point list."
    
    return wp_list[:, selected_j] 


def load_hscnn_R_model(model_dir, regress_mode, advanced_mode, crsval_mode):
    
    if advanced_mode['Physically_Plausible']:
        if advanced_mode['Data_Augmentation']:
            model_type = 'color_dataug'
        else:
            model_type = 'color'
    elif advanced_mode['Data_Augmentation']:
        model_type = 'dataug'
    else:
        model_type = 'orig'
    
    model_name =  'models_' + model_type + '_crsval_' + str(crsval_mode) + '/network_1/'
    print(model_dir + model_name + '*.pkl')
    model_path = glob(os.path.join(model_dir, model_name+'*.pkl'))
    assert len(model_path) == 1
    model_path = model_path[0] 

    save_point = torch.load(model_path)
    model_param = save_point['state_dict']
    
    for old_key in list(model_param.keys()): 
        new_key = 'module.'+old_key
        model_param[new_key] = model_param.pop(old_key)
    
    if advanced_mode['Physically_Plausible']:
        dim_spec = regress_mode['dim_spec'] - 3
    else:
        dim_spec = regress_mode['dim_spec']
    
    model = hscnn.resblock(hscnn.conv_relu_res_relu_block, 9, 3, dim_spec)
    
    model = nn.DataParallel(model)
    model.load_state_dict(model_param)
    
    model = model.cuda()
    model.eval()
    
    return model, model_type


def generate_file_name(mode, advanced_mode):
    
    if mode['type'] == 'poly':
        if mode['order'] == 1:
            gen_name = 'LR'
        else:
            gen_name = 'PR' + str(mode['order'])
    
    elif mode['type'] == 'root-poly':
        gen_name = 'RPR' + str(mode['order'])
    
    elif mode['type'] == 'rbf':
        gen_name = 'RBF'
    
    elif mode['type'] == 'HSCNN-R':
        gen_name = 'HSCNN-R'
    
    else:
        assert False, "No such regress_mode['type']"
    
    if advanced_mode['Data_Augmentation']:
        gen_name = gen_name + '_aug' + str(advanced_mode['Data_Augmentation'][0]) + '_' + str(advanced_mode['Data_Augmentation'][1])
    
    if advanced_mode['Sparse']:
        gen_name = gen_name + '_sparse'
    
    if advanced_mode['Physically_Plausible']:
        gen_name = gen_name + '_plausible'
    
    return gen_name


def generate_crsval_suffix(crsval_mode=0):
    if crsval_mode == 0:
        train_suffix = ""
        val_suffix = ""
        
    elif crsval_mode == 1:
        train_suffix = "_trainAB"
        val_suffix = "_valC"
        
    elif crsval_mode == 2:
        train_suffix = "_trainAB"
        val_suffix = "_valD"
    
    elif crsval_mode == 3:
        train_suffix = "_trainCD"
        val_suffix = "_valA"
    
    elif crsval_mode == 4:
        train_suffix = "_trainCD"
        val_suffix = "_valB"

    return train_suffix, val_suffix


def generate_crsval_imlist(crsval_name_list, crsval_mode):
    if crsval_mode == 1:
        train_list = crsval_name_list[0] + crsval_name_list[1]
        val_list = crsval_name_list[2]
        test_list = crsval_name_list[3]
    elif crsval_mode == 2:
        train_list = crsval_name_list[0] + crsval_name_list[1]
        val_list = crsval_name_list[3]
        test_list = crsval_name_list[2]
    elif crsval_mode == 3:
        train_list = crsval_name_list[2] + crsval_name_list[3]
        val_list = crsval_name_list[0]
        test_list = crsval_name_list[1]
    elif crsval_mode == 4:
        train_list = crsval_name_list[2] + crsval_name_list[3]
        val_list = crsval_name_list[1]
        test_list = crsval_name_list[0]
    
    return train_list, val_list, test_list
    
        
def write2csvfile(filename, row, edit_mode='a'):
    with open(filename, edit_mode, newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)
    file.close()


def initialize_csvfile(dir_result, regress_mode, advanced_mode, cost_funcs, test_modes):
    file_name = generate_file_name(regress_mode, advanced_mode) + '_' + str(datetime.date.today())    
    suffix = ''
    i = 0
    while os.path.isfile(os.path.join(dir_result, file_name + suffix + '.csv')):
        i += 1
        suffix = '_'+str(i)
    
    dir_name = os.path.join(dir_result, file_name + suffix + '.csv')
    
    row_A = ['Image']
    row_B = ['']
    row_C = ['']
    for exposure in advanced_mode['Exposure']:
        for cost_func in cost_funcs['test']:
            for tmode_key, tmode_func in test_modes.items():
                row_A.append('Exp ' + str(exposure))
                row_B.append(cost_func.__name__)
                row_C.append(tmode_key)
    
    write2csvfile(dir_name, row_A, edit_mode='w')
    write2csvfile(dir_name, row_B, edit_mode='a')
    write2csvfile(dir_name, row_C, edit_mode='a')
    
    return dir_name


def make_sure_dir_exist(directory):
    if os.path.isdir(directory):
        pass
    else:
        os.mkdir(directory)