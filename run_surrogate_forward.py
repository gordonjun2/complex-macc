# Copyright 2013-2018 Lawrence Livermore National Security, LLC and other
# @authors: :Rushil Anirudh, Jayaraman J. Thiagarajan, Timo Bremer
#
# SPDX-License-Identifier: MIT

import tensorflow as tf
import numpy as np
np.random.seed(4321)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from modelsv2 import *
from utils import *
import shutil
#import cPickle as pkl
import argparse
from sklearn.preprocessing import MinMaxScaler, scale

import wae_metric.model_AVB as wae
from wae_metric.utils import special_normalize
from wae_metric.run_WAE import LATENT_SPACE_DIM, load_dataset

import scipy.io

IMAGE_SIZE = 64

def run(**kwargs):
    infer_dir = kwargs.get('infer_dir', './surrogate_inference_results')
    modeldir = kwargs.get('modeldir','./surrogate_model_weights')
    ae_path = kwargs.get('ae_dir','./wae_metric/ae_model_weights')
    dataset = kwargs.get('dataset')
    complex_mode = kwargs.get('complex_mode')

    if dataset == 'fft-scattering-coef':
        complex_mode = True
        print('fft-scattering-coef data mode is selected...')
    else:
        print('icf-jag data mode is selected...')

    if complex_mode:
        print('Complex Mode is selected...')
    else:
        print('Non-complex Mode is selected...')

    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)

    if not os.path.exists(modeldir):
        print('Surrogate Model (Forward) Weight is not available! Please train the surrogate model first.')
        return

    if not os.path.exists(ae_path):
        print('Autoencoder Weight is not available! Please train the autoencoder first.')
        return

    input_sca = np.array(kwargs.get('input_sca'))
    input_sca = np.reshape(input_sca, (1, -1))

    # input_img = kwargs.get('input_img')

    # # Load input image (SKIPPED)

    # # Flatten input image (SKIPPED)
    # print('Flattening input image...')

    # flat_img = flatten_img(input_img)

    # print("Inputs dimensions: ", input_img.shape, input_sca.shape)

    # Concatenate input_img and input_sca (SKIPPED) 

    if dataset == 'fft-scattering-coef':
        dim_x = input_sca.shape[1]
        assert dim_x == 10, "There should be 10 input parameters."

        dim_y_img = 64 * 64 * 16
        dim_y_img_latent = LATENT_SPACE_DIM #latent space
        print("Input Size: ", dim_x)
        print("Output (Image) Size: ", dim_y_img)
        print("Latent Space Dimension: ", dim_y_img_latent)
    else:
        dim_x = input_sca.shape[1]
        assert dim_x == 5, "There should be 5 input parameters."

        dim_y_sca = 15
        dim_y_img = 64 * 64 * 4
        dim_y_img_latent = LATENT_SPACE_DIM #latent space
        print("Input Size: ", dim_x)
        print("Output (Image) Size: ", dim_y_img)
        print("Latent Space Dimension: ", dim_y_img_latent)

    print('Initialising model...')

    if complex_mode:
        x = tf.placeholder(tf.complex64, shape=[None, dim_x])
    else:
        x = tf.placeholder(tf.float32, shape=[None, dim_x])

    train_mode = tf.placeholder(tf.bool,name='train_mode')

    # y_mm = tf.concat([y_img,y_sca],axis=1)

    # Take out forward model from cycleGAN
    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''

    cycGAN_params_forward_only = {'input_params':x,
                                  'param_dim':dim_x,
                                  'output_dim':dim_y_img_latent,
                                  'L_adv':1e-2,
                                  'L_cyc':1e-1,
                                  'L_rec':1}

    JagNet_MM_forward_only = cycModel_MM_forward_only(**cycGAN_params_forward_only)
    JagNet_MM_forward_only.run(train_mode, complex_mode)

    # Take out decoder from autoencoder only
    '''**** Decode the prediction from latent vector --> img, scalars ****'''
    # Using the pretrained decoder from pretrained Multi-modal Wasserstein Autoencoder
    if dataset == 'fft-scattering-coef':
        y_img_out = wae.var_decoder_FCN(JagNet_MM_forward_only.output_fake, dim_y_img, train_mode = False, reuse = False, complex_mode = complex_mode)
    else:
        y_img_out = wae.var_decoder_FCN(JagNet_MM_forward_only.output_fake, dim_y_img+dim_y_sca, train_mode = False, reuse = False, complex_mode = complex_mode)

    # Separate weights for surrogate and decoder
    t_vars = tf.global_variables()
    m_vars = [var for var in t_vars if 'wae' in var.name]
    metric_saver = tf.train.Saver(m_vars)
    saver = tf.train.Saver(list(set(t_vars)-set(m_vars)))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(modeldir)
    ckpt_metric = tf.train.get_checkpoint_state(ae_path)

    if ckpt_metric and ckpt_metric.model_checkpoint_path:
        metric_saver.restore(sess, ckpt_metric.model_checkpoint_path)
        print("************ Image Metric Restored! **************")
    else:
        print("************ No Image Metric Found! **************")
        print('Please train the WAE first!')

        return

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")
    else:
        print("******* No Pretrained Model Found! ********")
        print('Please train the surrogate forward model first!')

        return

    print('Inference starts...')

    # Fix fd
    fd = {x: input_sca, train_mode:False}

    # Extract output
    pred_y, pred_x = sess.run([y_img_out,JagNet_MM_forward_only.input_cyc],
                                feed_dict=fd)
    
    # Output prediction

    inference_results(infer_dir, input_sca, pred_x, pred_y, complex_mode, dataset)
    
    # Save pred_y to .mat

    if dataset == 'fft-scattering-coef':
        pred_y_img = pred_y.reshape(-1,64,64,16)
        scipy.io.savemat(infer_dir + '/pred_fft_img.mat', {'ft_fft10': pred_y_img})



if __name__=='__main__':
    run()
