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

IMAGE_SIZE = 64

def run(**kwargs):
    fdir = kwargs.get('fdir', './surrogate_inference_results')
    modeldir = kwargs.get('modeldir','./surrogate_model_weights')
    ae_path = kwargs.get('ae_dir','./wae_metric/ae_model_weights')

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not os.path.exists(modeldir):
        print('Surrogate Model (Forward) Weight is not available! Please train the surrogate model first.')
        return

    if not os.path.exists(ae_path):
        print('Autoencoder Weight is not available! Please train the autoencoder first.')
        return

    complex_mode = kwargs.get('complex_mode')

    input_sca = kwargs.get('input_sca')
    input_img = kwargs.get('input_img')

    # Load input image (TODO)

    # Flatten input image (TODO)
    print('Flattening input image...')

    flat_img = flatten_img(input_img)

    print("Inputs dimensions: ", input_img.shape, input_sca.shape)

    # Concatenate input_img and input_sca (TODO) 

    dim_x = inp_img_sca.shape[1]
    dim_y_img_latent = LATENT_SPACE_DIM #latent space

    print('Initialising model...')

    if complex_mode:
        x = tf.placeholder(tf.complex64, shape=[None, dim_x])
    else:
        x = tf.placeholder(tf.float32, shape=[None, dim_x])

    train_mode = tf.placeholder(tf.bool,name='train_mode')

    # y_mm = tf.concat([y_img,y_sca],axis=1)

    # Take out forward model from cycleGAN (TODO)
    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''

    cycGAN_params = {'input_params':x,
                     'param_dim':dim_x,
                     'outputs':y_latent_img,
                     'output_dim':dim_y_img_latent,
                     'L_adv':1e-2,
                     'L_cyc':1e-1,
                     'L_rec':1}

    JagNet_MM = cycModel_MM(**cycGAN_params)
    JagNet_MM.run(train_mode, complex_mode)

    # Take out decoder from autoencoder only (TODO)
    '''**** Decode the prediction from latent vector --> img, scalars ****'''
    # Using the pretrained decoder from pretrained Multi-modal Wasserstein Autoencoder
    y_img_out = wae.var_decoder_FCN(JagNet_MM.output_fake, dim_y_img+dim_y_sca, train_mode = False, reuse = False, complex_mode = complex_mode)

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

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model restored! **************")

    print('Inference starts...')

    # Fix fd (TODO)
    fd = {x: x_mb, y_sca: y_sca_mb,y_img:y_img_mb,train_mode:True}

    # Extract output (TODO)
    gloss0,gloss1,gadv = sess.run([JagNet_MM.loss_gen0,
                                    JagNet_MM.loss_gen1,
                                    JagNet_MM.loss_adv],
                                    feed_dict=fd)

    # Print loss (TODO)
    if it_total % 100 == 0:
        print('Fidelity -- Iter: {}; Forward: {:.4f}; Inverse: {:.4f}'
            .format(it_total, gloss0, gloss1))
        print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}\n'.format(dloss,gadv))

    # Save output image (TODO)
    save_path = saver.save(sess, "./"+modeldir+"/model_"+str(it_total)+".ckpt")


if __name__=='__main__':
    run()

### srun -n 1 -N 1 -p pbatch -A lbpm --time=3:00:00 --pty /bin/sh
