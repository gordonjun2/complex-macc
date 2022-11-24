# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import numpy as np
import argparse
import run_cycgan_mm as cycGAN
import os
from wae_metric import run_WAE as metric

parser = argparse.ArgumentParser()
parser.add_argument('--train_ae', action='store_true',
                    help=' "-train_ae 0" to use pre-trained auto-encoder. "-train_ae 1": will train a new autoencoder before running the surrogate training.')
parser.add_argument('-d', type=str, default='icf-jag',
                    help='icf-jag or fft-scattering-coef, path to dataset - images, scalars, and input params')
parser.add_argument('--complex_mode', action='store_true',
                    help='option to use non-complex and complex images')
parser.add_argument('--split_n', default=4, type=int, 
                    help='split training data into split_n parts to reduce memory usage')       # use split_n = 1 when using fft dataset
parser.add_argument('--num_npy', default=10, type=int, 
                    help='load num_npy of fft dataset parts at one time to reduce memory usage')
parser.add_argument('--ae_batch_size', default=100, type=int, 
                    help='batch size for WAE training')
parser.add_argument('--forward_batch_size', default=64, type=int, 
                    help='batch size for forward surrogate training')

args = parser.parse_args()
dataset = args.d
complex_mode = args.complex_mode
split_n = args.split_n
num_npy = args.num_npy

ae_dir = './wae_metric/ae_model_weights'
ae_dir_outs = './wae_metric/ae_outs'

surrogate_dir = './surrogate_model_weights'
surrogate_dir_outs = './surrogate_outs'

if args.train_ae:
    batch_size = args.ae_batch_size
    print('****** Training the autoencoder *******')
    metric.run(fdir=ae_dir_outs,modeldir=ae_dir, dataset=dataset, complex_mode=complex_mode, split_n = split_n, num_npy = num_npy, batch_size = batch_size)
    # print('****** Training the macc surrogate *******')
    # # cycGAN.run(fdir,mdir,ae_dir)
    # cycGAN.run(fdir=surrogate_dir_outs,modeldir=surrogate_dir,ae_dir=ae_dir,dataset=dataset, complex_mode=complex_mode, split_n = split_n, num_npy = num_npy)
else:
    batch_size = args.forward_batch_size
    print('****** Training the macc surrogate with pre-trained autoencoder *******')
    cycGAN.run(fdir=surrogate_dir_outs,modeldir=surrogate_dir,ae_dir=ae_dir, dataset=dataset, complex_mode=complex_mode, split_n = split_n, num_npy = num_npy, batch_size = batch_size)
