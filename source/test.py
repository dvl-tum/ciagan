#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maximov
"""

import torch
import numpy as np
import util_func
import util_data
from torchvision import transforms, utils
from os.path import join
from os import listdir
import arch.arch_unet_flex as arch_gen
import argparse

def inference(generator, out_dir, data_loader, device_comp, num_classes = 1200):
    total_imgs = 0
    for batch in data_loader:
        # prepare data
        im_faces, im_lndm, im_msk, im_ind = [item[0].float().to(device_comp) for item in batch]

        output_id = (int(im_ind[0].cpu())+1)%num_classes #chose next id

        labels_one_hot = np.zeros((1, num_classes))
        labels_one_hot[0, output_id] = 1
        labels_one_hot = torch.tensor(labels_one_hot).float().to(device_comp)

        # inference
        with torch.no_grad():
            input_gen = torch.cat((im_lndm, im_faces * (1 - im_msk)), 1)
            im_gen = generator(input_gen, onehot=labels_one_hot)
            im_gen = torch.clamp(im_gen*im_msk+im_faces * (1 - im_msk), 0, 1) #final image with BG

        # output image
        img_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
        img_out.save(join(out_dir, str(total_imgs).zfill(6) + '.jpg'))
        total_imgs+=1
    
    print("Done.")



def run_inference(data_path='../dataset/celeba/', num_folders = -1, model_path = '../modelG', output_path = '../output'):
    ##### PREPARING DATA
    if num_folders==-1:
        num_folders = len(listdir(join(data_path,'lndm')))

    dataset_test = util_data.ImageDataset(root_dir=data_path, label_num=num_folders, transform_fnc=transforms.Compose([transforms.ToTensor()]), flag_sample=1, flag_augment = False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)

    ##### PREPARING MODELS
    device_comp = util_func.set_comp_device(True)
    model = arch_gen.Generator()
    model.load_state_dict(torch.load(model_path + '.pth'))
    model.to(device_comp)
    print('Model is ready')
    
    inference(model, output_path, data_loader, device_comp=device_comp)



# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to input data directory', default='../dataset/celeba/')
parser.add_argument('--ids', type=int, help='how many folder/ids to process', default=-1)
parser.add_argument('--model', type=str, help='path to a pre-trained model and its name', default='../modelG')
parser.add_argument('--out', type=str, help='path to output data directory', default='../output')

args = parser.parse_args()


run_inference(data_path=args.data, num_folders=args.ids,model_path=args.model, output_path=args.out)