#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, utils

import numpy as np
import importlib
import random
import math
from sacred import Experiment
import util_func as util_func
import util_loss as util_loss
import util_data as util_data

ciagan_exp = Experiment()

@ciagan_exp.config
def my_config():
    TRAIN_PARAMS = {
        'ARCH_NUM': 'unet',
        'ARCH_SIAM': 'unet',
        'FILTER_NUM': 32,
        'LEARNING_RATE': 0.0001,
        'FLAG_GPU': True,
        'EPOCHS_NUM': 100,
        'EPOCH_START': 0,

        'ITER_CRITIC': 5,
        'ITER_GENERATOR': 1,
        'ITER_SIAMESE':1,

        'GAN_TYPE':'lsgan',#wgangp
        'FLAG_SIAM_MASK': False,
    }

    DATA_PARAMS = {
        'DATA_PATH': '../dataset/',
        'DATA_SET': 'celeba',
        'LABEL_NUM': 1200,
        'WORKERS_NUM': 4,
        'BATCH_SIZE': 16,

        'IMG_SIZE': 128,
        'FLAG_DATA_AUGM': True,
    }

    OUTPUT_PARAMS = {
        'RESULT_PATH': '../results/',
        'MODEL_PATH': '../models/',
        'LOG_ITER': 50,
        'SAVE_EPOCH': 5,
        'SAVE_CHECKPOINT': 50,
        'VIZ_PORT': 8098, 'VIZ_HOSTNAME': "http://localhost", 'VIZ_ENV_NAME':'main',
        'SAVE_IMAGES': True,
        'PROJECT_NAME': 'ciagan',
        'EXP_TRY': 'check',
        'COMMENT': "Default",

    }

# sacred library
load_data = ciagan_exp.capture(util_data.load_data, prefix='DATA_PARAMS')
load_model = ciagan_exp.capture(util_func.load_model)
set_comp_device = ciagan_exp.capture(util_func.set_comp_device, prefix='TRAIN_PARAMS')
set_output_folders = ciagan_exp.capture(util_func.set_output_folders)
set_model_name = ciagan_exp.capture(util_func.set_model_name)

class Train_GAN():
    def __init__(self, model_info, device_comp, margin_contrastive=3, num_classes = 1200, gan_type = 'lsgan'):
        self.model_info = model_info
        self.device_comp = device_comp
        self.num_classes = num_classes
        self.gan_type = gan_type

        self.criterion_gan = util_loss.GANLoss(gan_type).to(self.device_comp)
        self.criterion_siamese = util_loss.ContrastiveLoss(margin_contrastive).to(self.device_comp)


    def save_model(self, epoch_iter=0, mode_save=0):
        if mode_save == 0:
            torch.save(self.model_info['generator'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_G.pth')
            torch.save(self.model_info['critic'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_C.pth')
            torch.save(self.model_info['siamese'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_S.pth')
        elif mode_save == 1:
            torch.save(self.model_info['generator'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'G.pth')
            torch.save(self.model_info['critic'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'C.pth')
            torch.save(self.model_info['siamese'].state_dict(), self.model_info['model_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + 'S.pth')

    def save_images(self, out, gt, target, inp, epoch_iter=0):
        viz_out_img = torch.clamp(out, 0., 1.)
        utils.save_image(viz_out_img, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_est.png")
        utils.save_image(gt, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_gt.png")
        utils.save_image(target, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep' + str(epoch_iter + 1).zfill(5) + "_tar.png")
        utils.save_image(inp, self.model_info['res_dir'] + self.model_info['model_name'] + '_ep'+ str(epoch_iter + 1).zfill(5) + "_inp.png")

    def reinit_loss(self):
        return [0] * 6, 0

    def process_batch_data(self, input_batch, flag_same = False):
        im_faces, im_lndm, im_msk, im_ind = input_batch

        im_faces = [item.float().to(self.device_comp) for item in im_faces]
        im_lndm = [item.float().to(self.device_comp) for item in im_lndm]
        im_msk = [item.float().to(self.device_comp) for item in im_msk]

        labels_one_hot = np.zeros((int(im_faces[0].shape[0]), self.num_classes))
        if len(im_ind)>1:
            labels_one_hot[np.arange(int(im_faces[1].shape[0])), im_ind[1]] = 1
        labels_one_hot = torch.tensor(labels_one_hot).float().to(self.device_comp)

        if flag_same:
            if len(im_ind) > 1:
                label_same = (im_ind[0]==im_ind[1])/1
            else:
                label_same = (im_ind[0]==im_ind[0])/1
            return im_faces, im_lndm, im_msk, labels_one_hot, label_same.to(self.device_comp)

        return im_faces, im_lndm, im_msk, labels_one_hot

    def train_siamese(self, num_iter_siamese=1):
        loss_sum = 0
        for p in self.model_info['siamese'].parameters():
            p.requires_grad = True

        for j in range(num_iter_siamese):
            # get and process new batch
            sample_batch = self.data_iter.next()
            im_faces, im_lndm, im_msk, im_onehot, label_data = self.process_batch_data(sample_batch, flag_same = True)

            # training part with full or masked image
            self.optimizer_S.zero_grad()
            fc_real1 = self.model_info['siamese'](im_faces[0] * (im_msk[0] if self.flag_siam_mask else 1))
            fc_real2 = self.model_info['siamese'](im_faces[1] * (im_msk[1] if self.flag_siam_mask else 1))
            loss_S = self.criterion_siamese(fc_real1, fc_real2, label_data)

            loss_S.backward()
            self.optimizer_S.step()
            loss_sum += loss_S.item()
        return loss_sum

    # Prepare inputs
    def input_train(self, im_faces, im_lndm, im_msk):
        input_repr = im_lndm
        input_gen = torch.cat((input_repr, im_faces[0] * (1 - im_msk[0])), 1)
        return input_gen, input_repr, im_msk

    def train_critic(self, num_iter_critic=1):
        loss_sum = [0,0]
        for p in self.model_info['critic'].parameters():
            p.requires_grad = True

        for j in range(num_iter_critic):
            # train region
            self.optimizer_C.zero_grad()

            # generate images from a frozen generator
            with torch.no_grad():
                sample_batch = self.data_iter.next()
                im_faces, im_lndm, im_msk, im_onehot = self.process_batch_data(sample_batch)

                input_gen, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)
                im_gen = self.model_info['generator'](input_gen, onehot = im_onehot)

            output_gen = im_faces[0] * (1 - im_msk[0]) + im_gen * im_msk[0]
            face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            loss_C_fake = self.criterion_gan(self.model_info['critic'](face_landmark_fake), False)

            # process real batch data
            sample_batch = self.data_iter.next()
            im_faces, im_lndm, im_msk, im_onehot = self.process_batch_data(sample_batch)

            output_real = im_faces[0]
            _, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)

            face_landmark_real = torch.cat((output_real, input_repr), 1)
            loss_C_real = self.criterion_gan(self.model_info['critic'](face_landmark_real), True)

            # update
            loss_D = loss_C_fake + loss_C_real

            if self.gan_type=='wgangp':
                grad_penalty = util_loss.cal_gradient_penalty(self.model_info['critic'], face_landmark_real, face_landmark_fake.detach(), self.device_comp)
                loss_D += grad_penalty

            loss_D.backward()
            self.optimizer_C.step()

            # for logging and visualization
            loss_sum[0] += loss_D.item()
            loss_sum[1] += loss_C_real.item()

        return loss_sum

    def train_generator(self, num_iter_generator=1, flag_siamese = True):
        loss_sum = [0,0]

        # freeze the discriminator
        for p in self.model_info['critic'].parameters():
            p.requires_grad = False
        for p in self.model_info['siamese'].parameters():
            p.requires_grad = False
        for p in self.model_info['generator'].parameters():
            p.requires_grad = True

        im_faces, im_lndm, output_gen = [],[],[]
        for j in range(num_iter_generator):
            # get and process data
            sample_batch = self.data_iter.next()
            im_faces, im_lndm, im_msk, im_onehot, label_same = self.process_batch_data(sample_batch, flag_same = True)

            # start training
            self.optimizer_G.zero_grad()
            input_gen, input_repr, im_msk = self.input_train(im_faces, im_lndm[0], im_msk)
            im_gen = self.model_info['generator'](input_gen, onehot=im_onehot)

            output_gen = im_faces[0] * (1 - im_msk[0]) + im_gen * im_msk[0]
            face_landmark_fake = torch.cat((output_gen, input_repr), 1)
            loss_G_gan = self.criterion_gan(self.model_info['critic'](face_landmark_fake), True)

            loss_G_siam = 0
            if flag_siamese:
                im_fake = output_gen
                fc_fake = self.model_info['siamese'](im_fake * (im_msk[0] if self.flag_siam_mask else 1))
                fc_real = self.model_info['siamese'](im_faces[1] * (im_msk[1] if self.flag_siam_mask else 1))
                labels_ones = torch.ones(label_same.shape).to(self.device_comp)
                loss_G_siam = self.criterion_siamese(fc_fake, fc_real, labels_ones)
                loss_sum[1] += loss_G_siam.item()

            loss_G = loss_G_gan + loss_G_siam
            loss_G.backward()
            self.optimizer_G.step()

            # for logging
            loss_sum[0] += loss_G_gan.item()
            for l_iter in range(len(label_same)):
                # alter colors depending on target class (red - different id, greenish - same id)
                if label_same[l_iter]==1:
                    im_faces[1][l_iter, 1, :, :] += 0.2
                    im_faces[1][l_iter, 2, :, :] += 0.2
                else:
                    im_faces[1][l_iter, 0, :, :] += 0.2

        return loss_sum, im_faces, im_lndm, output_gen

    @ciagan_exp.capture
    def train_model(self, loaders, TRAIN_PARAMS, OUTPUT_PARAMS):
        self.optimizer_G = optim.Adam(self.model_info['generator'].parameters(), lr=TRAIN_PARAMS['LEARNING_RATE'], betas=(0.5, 0.999))
        self.optimizer_C = optim.Adam(self.model_info['critic'].parameters(), lr=TRAIN_PARAMS['LEARNING_RATE'], betas=(0.5, 0.999))
        self.optimizer_S = optim.Adam(self.model_info['siamese'].parameters(), lr=TRAIN_PARAMS['LEARNING_RATE'], betas=(0.5, 0.999))

        self.flag_siam_mask = TRAIN_PARAMS['FLAG_SIAM_MASK']

        # Number of iterations per one training loop
        num_iter_siamese = TRAIN_PARAMS['ITER_SIAMESE']
        num_iter_critic = TRAIN_PARAMS['ITER_CRITIC']
        num_iter_generator = TRAIN_PARAMS['ITER_GENERATOR']
        self.model_info['total_steps'] = int(self.model_info['total_steps']/(2*num_iter_critic+num_iter_generator+num_iter_siamese))

        ##### Training
        print("Total number of epochs:", TRAIN_PARAMS['EPOCHS_NUM'])
        for e_iter in range(TRAIN_PARAMS['EPOCHS_NUM'] - TRAIN_PARAMS['EPOCH_START']):
            epoch_iter = e_iter + TRAIN_PARAMS['EPOCH_START']
            loss_sum, iter_count = self.reinit_loss()
            self.data_iter = iter(loaders[0])

            # Training procedure
            for st_iter in range(self.model_info['total_steps']):
                loss_sum[3] += self.train_siamese(num_iter_siamese=num_iter_siamese)

                loss_values = self.train_critic(num_iter_critic = num_iter_critic)
                loss_sum[0] += loss_values[0]
                loss_sum[4] += loss_values[1]

                train_out  = self.train_generator(num_iter_generator=num_iter_generator, flag_siamese=False if num_iter_siamese is 0 else True)
                loss_values, im_faces, im_lndm, im_gen = train_out[:4]
                loss_sum[1] += loss_values[0]
                loss_sum[2] += loss_values[1]
                iter_count += 1.

                ##### Log and visualize output
                if (st_iter + 1) % OUTPUT_PARAMS['LOG_ITER'] == 0:
                    print(self.model_info['model_name'], 'Epoch [{}/{}], Step [{}/{}], Loss C: {:.4f}, G: {:.4f}, S: {:.4f}'
                          .format(epoch_iter + 1, TRAIN_PARAMS['EPOCHS_NUM'], st_iter + 1, self.model_info['total_steps'], loss_sum[0] / iter_count, loss_sum[1] / iter_count, loss_sum[3] / iter_count))
                    total_iter = self.model_info['total_steps'] * epoch_iter + st_iter
                    loss_sum, iter_count = self.reinit_loss()

            ##### Save models and images
            if (epoch_iter + 1) % OUTPUT_PARAMS['SAVE_EPOCH']  == 0:
                self.save_model(mode_save=0)

                # make a separate checkpoint
                if (epoch_iter + 1) % OUTPUT_PARAMS['SAVE_CHECKPOINT'] == 0:
                    self.save_model(epoch_iter=e_iter, mode_save=1)



@ciagan_exp.automain
def run_exp(TRAIN_PARAMS):
    ##### INITIAL PREPARATIONS
    model_name = set_model_name()
    model_dir, res_dir = set_output_folders(model_name)
    device_comp = set_comp_device()

    ##### PREPARING DATA
    loader_train, total_steps, label_num = load_data(mode_train = True)
    loaders = [loader_train]

    ##### PREPARING MODELS
    ch_inp_num = 6
    generator = load_model(model_dir, model_name, 'Generator', device_comp, TRAIN_PARAMS['ARCH_NUM'], epoch_start = TRAIN_PARAMS['EPOCH_START'], ch_inp_num = ch_inp_num, label_num=label_num)
    critic = load_model(model_dir, model_name, 'Discriminator', device_comp, TRAIN_PARAMS['ARCH_NUM'], epoch_start = TRAIN_PARAMS['EPOCH_START'], ch_inp_num = ch_inp_num)

    if TRAIN_PARAMS['ARCH_SIAM'][:6]=='resnet':
        siamese = load_model(model_dir, model_name, 'ResNet', device_comp, TRAIN_PARAMS['ARCH_SIAM'], epoch_start=TRAIN_PARAMS['EPOCH_START'], ch_inp_num = 3)
    elif TRAIN_PARAMS['ARCH_SIAM'][:4] == 'siam':
        siamese = load_model(model_dir, model_name, 'NLayerDiscriminator', device_comp, TRAIN_PARAMS['ARCH_SIAM'], epoch_start=TRAIN_PARAMS['EPOCH_START'], ch_inp_num = 3)

    ##### PASSING INFO
    model_info = {'generator': generator,
                  'critic': critic,
                  'siamese': siamese,
                  'model_dir': model_dir,
                  'model_name': model_name,
                  'res_dir': res_dir,
                  'total_steps': total_steps,
                  'device_comp': device_comp,
                  'label_num': label_num,
                  }

    ##### INITIALIZE AND START TRAINING
    trainer = Train_GAN(model_info=model_info, device_comp=device_comp, num_classes=label_num, gan_type=TRAIN_PARAMS['GAN_TYPE'])
    trainer.train_model(loaders=loaders)
