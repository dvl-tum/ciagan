#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.utils.data

import os
from os import listdir, mkdir
from os.path import isfile, join, isdir
import numpy as np
import importlib

def weights_init(m):
    if type(m) == nn.Conv2d:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.xavier_normal(m.weight)
        m.bias.data.fill_(0.01)


def set_comp_device(FLAG_GPU):
    ##### Device configuration
    device_comp = torch.device("cpu")
    if FLAG_GPU:
        device_comp = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device_comp


def set_model_name(OUTPUT_PARAMS, DATA_PARAMS, TRAIN_PARAMS):
    model_name = 'A' + str(TRAIN_PARAMS['ARCH_NUM']).zfill(2) + '_D' + str(DATA_PARAMS['DATA_SET']).zfill(2) + '_T' + \
                 str(OUTPUT_PARAMS['EXP_TRY']).zfill(2)
    return model_name


def load_model(model_dir, model_name, arch_name, device_comp, arch_type, DATA_PARAMS, epoch_start = 0, ch_inp_num = 6, label_num=1200):
    arch = importlib.import_module('arch.arch_' + str(arch_type))

    model = getattr(arch, arch_name)(input_nc=ch_inp_num, num_classes=label_num, encode_one_hot = True, img_size = DATA_PARAMS['IMG_SIZE'])
    if arch_name=='Generator':
        model.apply(weights_init)

    if epoch_start > 0:
        model.load_state_dict(torch.load(model_dir + model_name + '_ep' + str(epoch_start) + '.pth'))
        print("Model loaded:", model_name, " epoch:", str(epoch_start))
    model = model.to(device=device_comp)

    return model


def set_output_folders(model_name, OUTPUT_PARAMS):
    res_dir = OUTPUT_PARAMS['RESULT_PATH'] + OUTPUT_PARAMS['PROJECT_NAME'] + '_' + model_name + '/'
    models_dir = OUTPUT_PARAMS['MODEL_PATH'] + OUTPUT_PARAMS['PROJECT_NAME'] + '_' + model_name + '/'
    if not isdir(OUTPUT_PARAMS['RESULT_PATH']):
        mkdir( OUTPUT_PARAMS['RESULT_PATH'])
    if not isdir( OUTPUT_PARAMS['MODEL_PATH']):
        mkdir( OUTPUT_PARAMS['MODEL_PATH'])

    if not isdir(models_dir):
        mkdir(models_dir)
    if not isdir(res_dir):
        mkdir(res_dir)
    return models_dir, res_dir
