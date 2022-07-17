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
parser.add_argument('-o', type=str, default='out',
                    help='Saving Results Directory')
parser.add_argument('-m', type=str, default='weights',
                    help='location to store weights')
parser.add_argument('--train_ae', action='store_true',
                    help=' "-train_ae 0" to use pre-trained auto-encoder. "-train_ae 1": will train a new autoencoder before running the surrogate training.')
parser.add_argument('-ae_dir', type=str, default='./wae_metric/model_weights',
                    help='Ignored if train_ae=True; else will load existing autoencoder')
parser.add_argument('-d', type=str, default='./data/icf-jag-10k/',
                    help='path to dataset - images, scalars, and input params')
parser.add_argument('--complex_mode', action='store_true',
                    help='option to use non-complex and complex images')

args = parser.parse_args()
fdir = args.o
mdir = args.m
ae_dir = args.ae_dir
datapath = args.d
complex_mode = args.complex_mode

ae_dir = 'wae_metric/ae_model_'+mdir
ae_dir_outs = 'wae_metric/ae_outs'

surrogate_dir = './surrogate_model_weights'
surrogate_dir_outs = './surrogate_outs'

if args.train_ae:
    print('****** Training the autoencoder *******')
    metric.run(fdir=ae_dir_outs,modeldir=ae_dir,datapath=datapath, complex_mode=complex_mode)
    print('****** Training the macc surrogate *******')
    # cycGAN.run(fdir,mdir,ae_dir)
    cycGAN.run(fdir=surrogate_dir_outs,modeldir=surrogate_dir,ae_dir=ae_dir,datapath=datapath, complex_mode=complex_mode)
else:
    print('****** Training the macc surrogate with pre-trained autoencoder *******')
    cycGAN.run(fdir=surrogate_dir_outs,modeldir=surrogate_dir,ae_dir=ae_dir,datapath=datapath, complex_mode=complex_mode)
