# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import numpy as np
import argparse
import run_cycgan_mm as cycGAN
import os
from wae_metric import run_WAE as metric

# Arguments for running this script (settings for training)
# Note: Some arguments requires value, while some arguments are just flags.
parser = argparse.ArgumentParser()

# Using '--train_ae' flag switches training for Autoencoder, while not using it switches training for Surrogate Forward and Inverse Model (Note that a 
# pretrained autoencoder model is required for training the Surrogate Forward and Inverse Model). 
parser.add_argument('--train_ae', action='store_true',
                    help=' "-train_ae 0" to use pre-trained auto-encoder. "-train_ae 1": will train a new autoencoder before running the surrogate training.')

# '-d' sets the dataset to use. To use the inertial confinement fusion simulation dataset, set value as 'icf-jag'. To use the FT scattering coefficient 
# dataset, set value as 'fft-scattering-coef'.
parser.add_argument('-d', type=str, default='icf-jag',
                    help='icf-jag or fft-scattering-coef, path to dataset - images, scalars, and input params')

# With '--complex_mode' flag enabled, the complex version of a dataset will be used. This is only relevant to 'icf-jag' dataset as 'fft-scattering-coef' 
# dataset is already complex-valued. 
parser.add_argument('--complex_mode', action='store_true',
                    help='option to use non-complex and complex images')

# [ADDED] '--split_n' divides the dataset into 'n' parts before loading the 'n' parts separately during the training. This helps to reduce memory usage.
# For example, the dataset contains 1,000,000 data. By default, all data will be loaded at once. This requires high memory usage. If n = 10, then only 
# 1,000,000 / 10 = 100,000 data will be loaded at once. During the training, the rest of the data in a set of 100,000 will be loaded separately. If n = 1, 
# the dataset will not be divided.  
parser.add_argument('--split_n', default=4, type=int, 
                    help='split training data into split_n parts to reduce memory usage')        # use split_n = 1 when using fft dataset

# [ADDED] '--num_npy' loads 'n' number of 'fft-scattering-coef' at a time during the training. This helps to reduce memory usage. For example, there are
# 40,000 .npy files for the 'fft-scattering-coef' dataset. Each .npy file will contain a certain number of data (eg. 100). It will be memory intensive to 
# load all 40,000 .npy files or 40,000 .npy files x 100 data = 4,000,000 data at the same time. If n = 10, then only 10 .npy files or 10 .npy files x 100 
# data = 1,000 data will be loaded at once. During the training, the rest of the data in a set of 10 .npy files will be loaded separately. If n = 1, all 
# .npy files will be loaded at once. 
parser.add_argument('--num_npy', default=10, type=int, 
                    help='load num_npy of fft dataset parts at one time to reduce memory usage') # for testing purpose, use 2 first

# '--ae_batch_size' sets the batch size for the Autoencoder's training (preferably as high as the PC can handle (generally) and keeping the batch size 
# value to a power of 2, eg. 4, 8, 16, 32, 64, etc)
parser.add_argument('--ae_batch_size', default=100, type=int, 
                    help='batch size for WAE training')

# '--forward_batch_size' sets the batch size for the Surrogate Forward and Inverse Model's training (preferably as high as the PC can handle and keeping 
# the batch size value to a power of 2, eg. 4, 8, 16, 32, 64, etc)
parser.add_argument('--forward_batch_size', default=64, type=int, 
                    help='batch size for forward surrogate training')

args = parser.parse_args()
dataset = args.d
complex_mode = args.complex_mode
split_n = args.split_n
num_npy = args.num_npy

# 'ae_dir' is the directory where the trained Autoencoder models are saved.
ae_dir = './wae_metric/ae_model_weights'

# 'ae_dir_outs' is the directory where training results (visualisation) for Autoencoder are saved.
ae_dir_outs = './wae_metric/ae_outs'

# 'surrogate_dir' is the directory where the trained Surrogate Forward and Inverse models are saved.
surrogate_dir = './surrogate_model_weights'

# 'surrogate_dir_outs' is the directory where training results (visualisation) for Surrogate Forward and Inverse Model are saved.
surrogate_dir_outs = './surrogate_outs'

if args.train_ae:
    batch_size = args.ae_batch_size
    print('****** Training the autoencoder *******')
    # Continues at './wae_metric/run_WAE.py'
    metric.run(fdir=ae_dir_outs,modeldir=ae_dir, dataset=dataset, complex_mode=complex_mode, split_n = split_n, num_npy = num_npy, batch_size = batch_size)
else:
    batch_size = args.forward_batch_size
    print('****** Training the macc surrogate with pre-trained autoencoder *******')
    # Continues at './run_cycgan_mm.py.py'
    cycGAN.run(fdir=surrogate_dir_outs,modeldir=surrogate_dir,ae_dir=ae_dir, dataset=dataset, complex_mode=complex_mode, split_n = split_n, num_npy = num_npy, batch_size = batch_size)
