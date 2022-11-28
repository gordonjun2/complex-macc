# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import numpy as np
import argparse
import run_cycgan_mm as cycGAN
import run_surrogate_forward as surrogateForward
import os
from wae_metric import run_WAE as metric

# Arguments for running this script (settings for training)
# Note: Some arguments requires value, while some arguments are just flags.
parser = argparse.ArgumentParser()

# '-ae_dir' sets the directory where the Autoencoder's pretrained weights can be found.
parser.add_argument('-ae_dir', type=str, default='./wae_metric/ae_model_weights',
                    help='Autoencoder weight')

# '-cyc_dir' sets the directory where the Surrogate Forward and Inverse Model's pretrained weights can be found.
parser.add_argument('-cyc_dir', type=str, default='./surrogate_model_weights',
                    help='Surrogate (Forward) and Inverse Neural Network (Inverse) weight')

# '-d' sets the dataset to use. To use the inertial confinement fusion simulation dataset, set value as 'icf-jag'. To use the FT scattering coefficient 
# dataset, set value as 'fft-scattering-coef'.
parser.add_argument('-d', type=str, default='icf-jag',
                    help='icf-jag or fft-scattering-coef, path to dataset - images, scalars, and input params')

# parser.add_argument('-input_sca', type=float, default=[],
#                     help='Input parameters for the simulation')                             # input parameters as list, e.g. [1, 2, 3, 4, ..]
# parser.add_argument('-input_img', type=str, default='./sample_images/<image.jpg>',
#                     help='load a single input image')                                       # NOT BEING IMPLEMENTED YET

# With '--complex_mode' flag enabled, the complex version of a dataset will be used. This is only relevant to 'icf-jag' dataset as 'fft-scattering-coef' 
# dataset is already complex-valued. 
parser.add_argument('--complex_mode', action='store_true',
                    help='option to use non-complex and complex images')

args = parser.parse_args()
ae_dir = args.ae_dir
cyc_dir = args.cyc_dir
dataset = args.d

# Type scalar input parameters here
input_sca = [-0.07920084, 0.70821885, 0.377287, 0.12390906, 0.22148967]
# input_img = args.input_img

complex_mode = args.complex_mode

# 'surrogate_dir_outs' is the directory where inference results (visualisation) are saved.
surrogate_dir_outs = './surrogate_inference_results'

print('****** Simulating output from input using macc surrogate (forward) *******')
# Continues at './run_surrogate_forward.py'
surrogateForward.run(infer_dir=surrogate_dir_outs,modeldir=cyc_dir,ae_dir=ae_dir, dataset=dataset, complex_mode=complex_mode, input_sca = input_sca)
