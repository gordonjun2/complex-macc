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
from wae_metric.run_WAE import LATENT_SPACE_DIM, load_dataset, load_fft_dataset_actual

import math

# 'IMAGE_SIZE' is kept as 64 for both 'icf-jag' and 'fft-scattering-coef' dataset.
IMAGE_SIZE = 64

def run(**kwargs):
    # Retrieves the argument variables in './main.py'
    fdir = kwargs.get('fdir', './surrogate_outs')
    modeldir = kwargs.get('modeldir','./surrogate_model_weights')
    ae_path = kwargs.get('ae_dir','./wae_metric/ae_model_weights')
    dataset = kwargs.get('dataset')
    batch_size = kwargs.get('batch_size')

    visdir = './tensorboard_plots'
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    if not os.path.exists(visdir):
        os.makedirs(visdir)

    complex_mode = kwargs.get('complex_mode')
    split_n = kwargs.get('split_n')
    num_npy = kwargs.get('num_npy')

    # Selecting dataset as 'fft-scattering-coef' forces the 'complex_mode' to switch on
    if dataset == 'fft-scattering-coef':
        complex_mode = True
        print('fft-scattering-coef dataset is selected...')
    else:
        print('icf-jag dataset is selected...')

    if complex_mode:
        print('Complex Mode is selected...')
    else:
        print('Non-complex Mode is selected...')

    print('Loading dataset...')

    # To read dataset directory / load dataset and get randomised training and testing data indexes to be used in training later 
    if dataset == 'fft-scattering-coef':
        
        fft_datapath = './data/fft-scattering-coef-40k/'

        # For Surrogate Forward and Inverse Model's training, both images (for Autoencoder part) and scalar inputs (for Surrogate Forward and Inverse Model) part) are used.
        # Read the dataset directory first (NO LOADING yet)

        fft_img_datapath = fft_datapath + 'fft40K_images/'
        fft_img_files_list = os.listdir(fft_img_datapath)

        fft_inp_datapath = fft_datapath + 'fft40K_params/'
        fft_inp_files_list = os.listdir(fft_inp_datapath)

        fft_data_size_per_npy_set = num_npy * 100                   # Each npy has 100 data, thus total data is num_npy * 100

        # Randomised training and testing data indexes to be used in training later (to prevent training bias in ordering)
        # For example, there are 10 data. Let no. of training data be 8 and no. of testing data be 2. 
        # A randomised 'tr_id' can be [0, 8, 4, 5, 6, 7, 3, 2], while a randomised 'te_id' is [1, 9].
        tr_id = np.random.choice(fft_data_size_per_npy_set,int(fft_data_size_per_npy_set*0.95),replace=False)
        te_id = list(set(range(fft_data_size_per_npy_set)) - set(tr_id))

        # A sample .npy is loaded in order to plot the sample ground truth data below
        fft_img_files_list_temp = ['fft40K_images_1.npy']
        fft_img_temp = load_fft_dataset_actual(fft_img_datapath, fft_img_files_list_temp)
        
        fft_inp_files_list_temp = ['fft40K_params_1.npy']
        fft_inp_temp = load_fft_dataset_actual(fft_inp_datapath, fft_inp_files_list_temp)

        y_img_temp = fft_img_temp
        y_img_temp_mb = y_img_temp[-16:,:]
        y_img_temp_mb = y_img_temp_mb.reshape(16,64,64,16)

        x_inp_temp = fft_inp_temp

    else:
        # Unlike 'fft-scattering-coef', the data in 'icf-jag' is fully loaded to memory first (but not to training).
        jag_inp, jag_sca, jag_img = load_dataset(complex_mode)

        # Randomised training and testing data indexes to be used in training later (to prevent training bias in ordering)
        # For example, there are 10 data. Let no. of training data be 8 and no. of testing data be 2. 
        # A randomised 'tr_id' can be [0, 8, 4, 5, 6, 7, 3, 2], while a randomised 'te_id' is [1, 9].
        tr_id = np.random.choice(jag_sca.shape[0],int(jag_sca.shape[0]*0.95),replace=False)
        te_id = list(set(range(jag_sca.shape[0])) - set(tr_id))

        if complex_mode:
            X_test = np.complex64(jag_inp[te_id,:])
            y_sca_test = np.complex64(jag_sca[te_id,:])
        else:
            X_test = jag_inp[te_id,:]
            y_sca_test = jag_sca[te_id,:]
        y_img_test = jag_img[te_id,:]
        y_img_test_mb = y_img_test[-16:,:]

        y_img_test_mb = y_img_test_mb.reshape(16,64,64,4)
    
    print('Splitting dataset indexes...')

    # The dataset is splitted up into 'split_n' parts according to the argument variable used in './main.py'.
    # 'sub_X_train_len' is the training length per 'split_n' parts. For example (using the example above), there are 8 training 
    # data. If split_n = 2, sub_X_train_len = 8 / 2 = 4. Then tr_id = [0, 8, 4, 5, 6, 7, 3, 2] will be splitted into tr_id_split 
    # = [[0, 8, 4, 5], [6, 7, 3, 2]]. As a result, indexes [0, 8, 4, 5] will be loaded first, then it will be replaced by 
    # indexes [6, 7, 3, 2].
    sub_X_train_len = len(tr_id)/split_n

    if sub_X_train_len % 1 == 0:
        # Enters if 'sub_X_train_len' is an integer (no remainder)
        sub_X_train_len = int(sub_X_train_len)
        tr_id_split = [tr_id[i*sub_X_train_len:i*sub_X_train_len+sub_X_train_len] for i in range(split_n)]
    else:
        # Enters if 'sub_X_train_len' is not an integer (got remainder, so the last set of indexes is shorter than 'sub_X_train_len')
        sub_X_train_len = len(tr_id)//split_n
        tr_id_split = [tr_id[i*sub_X_train_len:i*sub_X_train_len+sub_X_train_len] if i < sub_X_train_len else tr_id[i*sub_X_train_len:] for i in range(split_n+1)]

    # 'batch_size' cannot be more than 'sub_X_train_len'(cannot process more than what is there). Thus, if 'batch_size' is larger 
    # than 'sub_X_train_len', use 'sub_X_train_len' as batch_size instead.
    if batch_size > sub_X_train_len:
        batch_size = sub_X_train_len
        print('Selected batch size is larger than splitted batch size... Using splitted batch size instead...')

    print('Batch Size: ', batch_size)

    # Plot sample ground truth data
    if dataset == 'fft-scattering-coef':
        ks = 16

        for k in range(ks):
            # Real
            fig = plot(np.real(y_img_temp_mb[:,:,:,k]),immax=np.max(np.real(y_img_temp_mb[:,:,:,k]).reshape(-1,4096),axis=1),
                    immin=np.min(np.real(y_img_temp_mb[:,:,:,k]).reshape(-1,4096),axis=1))
            plt.savefig('{}/gt_real_img_{}_{}.png'
                        .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
            plt.close()

            # Imaginary
            fig = plot(np.imag(y_img_temp_mb[:,:,:,k]),immax=np.max(np.imag(y_img_temp_mb[:,:,:,k]).reshape(-1,4096),axis=1),
                    immin=np.min(np.imag(y_img_temp_mb[:,:,:,k]).reshape(-1,4096),axis=1))
            plt.savefig('{}/gt_imag_img_{}_{}.png'
                        .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
            plt.close()

    else:
        ks = 4
        
        for k in range(ks):
            if complex_mode:
                # Real
                fig = plot(np.real(y_img_test_mb[:,:,:,k]),immax=np.max(np.real(y_img_test_mb[:,:,:,k]).reshape(-1,4096),axis=1),
                        immin=np.min(np.real(y_img_test_mb[:,:,:,k]).reshape(-1,4096),axis=1))
                plt.savefig('{}/gt_real_img_{}_{}.png'
                            .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
                plt.close()

                # Imaginary
                fig = plot(np.imag(y_img_test_mb[:,:,:,k]),immax=np.max(np.imag(y_img_test_mb[:,:,:,k]).reshape(-1,4096),axis=1),
                        immin=np.min(np.imag(y_img_test_mb[:,:,:,k]).reshape(-1,4096),axis=1))
                plt.savefig('{}/gt_imag_img_{}_{}.png'
                            .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
                plt.close()

            else:
                fig = plot(y_img_test_mb[:,:,:,k],immax=np.max(y_img_test_mb[:,:,:,k].reshape(-1,4096),axis=1),
                        immin=np.min(y_img_test_mb[:,:,:,k].reshape(-1,4096),axis=1))
                plt.savefig('{}/gt_img_{}_{}.png'
                            .format(fdir,str(k).zfill(3),str(k)), bbox_inches='tight')
                plt.close()

    if dataset == 'fft-scattering-coef':
        print("Dataset dimensions: ", x_inp_temp[-16:,:].shape, y_img_temp[-16:,:].shape)
        dim_x = x_inp_temp.shape[1]
        dim_y_img = y_img_temp.shape[1]
        dim_y_img_latent = LATENT_SPACE_DIM #latent space
    else:
        print("Dataset dimensions: ",X_test.shape,y_sca_test.shape,y_img_test.shape)
        dim_x = jag_inp.shape[1]
        dim_y_sca = jag_sca.shape[1]
        dim_y_img = jag_img.shape[1]
        dim_y_img_latent = LATENT_SPACE_DIM #latent space

    ### Metric params

    # ''' TEST mini-batch '''
    # x_test_mb = X_test[-100:,:]
    # y_sca_test_mb = y_sca_test[-100:,:]
    # y_img_test_mb = y_img_test[-100:,:]

    print('Initialising model...')

    # Tensorflow placeholder and function initialisation stage
    # Note: In Tensorflow, a computational graph has to be set up before data for training can be fit into it. The 
    # computational graph is created based on the data size and the selected network size. A simple analogy would 
    # be a subway system (might not be inaccurate..):
    # - Purchase a train model (data)
    # - Create railway tunnels and tracks based on the train model (neural network)
    # - Once the railway tunnels and tracks are created, the train can finally be deployed (data --> neural network 
    # model) 
    #
    # Also, the data type of the graph will also be initialised according to the data type of the input. If 
    # 'complex_mode' is on, 'complex64' will be used, else, 'float32' is used.

    if complex_mode:
        if dataset != 'fft-scattering-coef':
            y_sca = tf.placeholder(tf.complex64, shape=[None, dim_y_sca])
        y_img = tf.placeholder(tf.complex64, shape=[None, dim_y_img])
        x = tf.placeholder(tf.complex64, shape=[None, dim_x])
    else:
        y_sca = tf.placeholder(tf.float32, shape=[None, dim_y_sca])
        y_img = tf.placeholder(tf.float32, shape=[None, dim_y_img])
        x = tf.placeholder(tf.float32, shape=[None, dim_x])

    train_mode = tf.placeholder(tf.bool,name='train_mode')

    if dataset == 'fft-scattering-coef':
        y_mm = y_img
    else:
        y_mm = tf.concat([y_img,y_sca],axis=1)

    '''**** Encode the img, scalars ground truth --> latent vector for loss computation ****'''
    # Using the pretrained encoder from pretrained Multi-modal Wasserstein Autoencoder
    y_latent_img = wae.gen_encoder_FCN(y_mm, dim_y_img_latent, train_mode = False, reuse = False, complex_mode = complex_mode)

    '''**** Train cycleGAN input params <--> latent space of (images, scalars) ****'''

    cycGAN_params = {'input_params':x,
                     'param_dim':dim_x,
                     'outputs':y_latent_img,
                     'output_dim':dim_y_img_latent,
                     'L_adv':1e-2,
                     'L_cyc':1e-1,
                     'L_rec':1}                 # initial L_adv = 1e-2, initial L_cyc = 1e-1, initial L_rec = 1

    JagNet_MM = cycModel_MM(**cycGAN_params)
    JagNet_MM.run(train_mode, complex_mode)

    '''**** Decode the prediction from latent vector --> img, scalars ****'''
    # Using the pretrained decoder from pretrained Multi-modal Wasserstein Autoencoder
    if dataset == 'fft-scattering-coef':
        y_img_out = wae.var_decoder_FCN(JagNet_MM.output_fake, dim_y_img, train_mode = False, reuse = False, complex_mode = complex_mode)
    else:
        y_img_out = wae.var_decoder_FCN(JagNet_MM.output_fake, dim_y_img+dim_y_sca, train_mode = False, reuse = False, complex_mode = complex_mode)

    # Note: 16384 = 64 x 64 x 4 ('icf-jag' image has a resolution of 64 x 64 pixel size with 4 channels.).
    if complex_mode:
        if dataset == 'fft-scattering-coef':
            img_loss = tf.reduce_mean(tf.square(tf.abs(y_img_out - y_img)))
        else:
            img_loss = tf.reduce_mean(tf.square(tf.abs(y_img_out[:,:16384] - y_img)))
            sca_loss = tf.reduce_mean(tf.square(tf.abs(y_img_out[:,16384:] - y_sca)))
    else:
        img_loss = tf.reduce_mean(tf.square(y_img_out[:,:16384] - y_img))
        sca_loss = tf.reduce_mean(tf.square(y_img_out[:,16384:] - y_sca))

    fwd_img_summary = tf.summary.scalar(name='Image Loss', tensor=img_loss)
    if dataset != 'fft-scattering-coef':
        fwd_sca_summary = tf.summary.scalar(name='Scalar Loss', tensor=sca_loss)
    merged = tf.summary.merge_all()

    t_vars = tf.global_variables()
    m_vars = [var for var in t_vars if 'wae' in var.name]
    metric_saver = tf.train.Saver(m_vars)
    saver = tf.train.Saver(list(set(t_vars)-set(m_vars)))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(modeldir)
    ckpt_metric = tf.train.get_checkpoint_state(ae_path)

    # 'ckpt_metric.model_checkpoint_path' checks and reads the 'checkpoint' file in './wae_metric/ae_model_weights'.
    # A pretrained Autoencoder is required. If it does not exist, the program will stop.
    if ckpt_metric and ckpt_metric.model_checkpoint_path:
        metric_saver.restore(sess, ckpt_metric.model_checkpoint_path)
        print("************ Image Metric Restored! **************")
    else:
        print("************ No Image Metric Found! **************")
        print('Please train the WAE first!')

        return

    # 'ckpt.model_checkpoint_path' checks and reads the 'checkpoint' file in './surrogate_model_weights'.
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model Restored! **************")
    else:
        print("******* No Pretrained Model Found! ********")

    writer = tf.summary.FileWriter(visdir+'/{}'.format(modeldir), sess.graph)

    print('Training starts...')

    it_total = 0
    if dataset == 'fft-scattering-coef':
        it_max = 400000
    else:
        it_max = 100000

    print('A total of ' + str(it_max) + ' data iteration is selected for the training...')

    # Training starts here

    if dataset == 'fft-scattering-coef':

        # 'fft_img_files_list' and 'fft_inp_files_list' are lists that contains all the .npy file names for both images and scalar 
        # inputs. 'math.ceil(len(fft_img_files_list)/num_npy)' refers to the no. of time a set of .npy files is being loaded. For 
        # example, len(fft_img_files_list) = 10 and num_npy = 2. Then, math.ceil(len(fft_img_files_list)/num_npy) = 10 / 2 = 5. 
        # This means 2 .npy files will be loaded per 'for' loop, and there will be 5 loops in total. For the purpose of being 
        # general, 'math.ceil' rounds the division up in case len(fft_img_files_list)/num_npy is not integer. The same is applied 
        # to 'fft_inp_files_list'.
        for i in range(math.ceil(len(fft_img_files_list)/num_npy)):
            fft_img_files_list_split = fft_img_files_list[i*num_npy:i*num_npy+num_npy]
            fft_inp_files_list_split = fft_inp_files_list[i*num_npy:i*num_npy+num_npy]

            # 'fft_img' and 'fft_inp' are the actual data used to train the neural network
            fft_img = load_fft_dataset_actual(fft_img_datapath, fft_img_files_list_split)
            fft_inp = load_fft_dataset_actual(fft_inp_datapath, fft_inp_files_list_split)

            # While 'fft_img' and 'fft_inp' are being loaded to memory (RAM), they are not loaded into the GPU. The data is further 
            # splitted according to 'tr_id_split' (see above).
            for tr_id in tr_id_split:
                X_train = np.complex64(fft_inp[tr_id,:])
                y_img_train = fft_img[tr_id,:]

                X_test = np.complex64(fft_inp[te_id,:])
                y_img_test = fft_img[te_id,:]

                # '(it_max//math.ceil(len(fft_img_files_list)/num_npy))//split_n' refers to the number of iteration a set of data is 
                # being used for training. As established above, 'math.ceil(len(fft_img_files_list)/num_npy)' refers to the no. of 
                # time a set of .npy files is being loaded. Then 'it_max//math.ceil(len(fft_img_files_list)/num_npy)' refers to the 
                # no. of training iteration per a set of .npy files that is being loaded. Then for every set of .npy files that is 
                # being loaded, only a subset of data will be loaded onto the GPU for training at once. This is governed by the 
                # 'split_n' (see above). Thus, with '(it_max//math.ceil(len(fft_img_files_list)/num_npy))//split_n', every subset of 
                # data will be given equal (statistically) no. of chance to be trained on.
                for it in range(0, (it_max//math.ceil(len(fft_img_files_list)/num_npy))//split_n):

                    if X_train.shape[0] < batch_size:
                        batch_size = X_train.shape[0]

                    # The training indexes per subset of data is randomised
                    randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
                    x_mb = X_train[randid,:]
                    y_img_mb = y_img_train[randid,:]
                    
                    # Feed dictionary for training (or the inputs to the training model)
                    fd = {x: x_mb,y_img:y_img_mb,train_mode:True}

                    for _ in range(10):
                        _,dloss = sess.run([JagNet_MM.D_solver,JagNet_MM.loss_disc],feed_dict=fd)

                    gloss0,gloss1,gadv = sess.run([JagNet_MM.loss_gen0,
                                                    JagNet_MM.loss_gen1,
                                                    JagNet_MM.loss_adv],
                                                    feed_dict=fd)
                    
                    for _ in range(1):
                        _ = sess.run([JagNet_MM.G0_solver],feed_dict=fd)

                    if (it_total+1) % 100 == 0:
                        print('Fidelity -- Iter: {}; Forward: {:.4f}; Inverse: {:.4f}'
                            .format(it_total+1, gloss0, gloss1))
                        if (it_total+1) % 500 == 0:
                            print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}'.format(dloss,gadv))
                        else:
                            print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}\n'.format(dloss,gadv))


                    if (it_total+1) % 500 == 0:
                        summary_val = sess.run(merged,feed_dict={x:X_test,y_img:y_img_test,train_mode:False})

                        writer.add_summary(summary_val, it_total+1)

                        nTest=16
                        x_temp_mb = x_inp_temp[-nTest:,:]
                        y_img_temp_mb = y_img_temp[-nTest:,:]

                        samples,samples_x = sess.run([y_img_out,JagNet_MM.input_cyc],
                                                    feed_dict={x: x_temp_mb,train_mode:False})
                        data_dict= {}
                        data_dict['samples'] = samples
                        data_dict['samples_x'] = samples_x
                        data_dict['y_img'] = y_img_temp_mb
                        data_dict['x'] = x_temp_mb

                        # Saves testing results in './surrogate_outs'.
                        test_imgs_plot(fdir,it_total,data_dict, complex_mode, dataset)

                    # Saves trained model in './surrogate_model_weights'.
                    # Can set which iteration should the model be saved
                    if (it_total+1)%(it_max//100)==0:
                        if complex_mode:
                            save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_complex_model_"+str(it_total)+".ckpt")
                        else:
                            save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_model_"+str(it_total)+".ckpt")

                    it_total = it_total + 1

    else:
        # The data is splitted according to 'tr_id_split' (see above).
        for tr_id in tr_id_split:
            if complex_mode:
                X_train = np.complex64(jag_inp[tr_id,:])
                y_sca_train = np.complex64(jag_sca[tr_id,:])
            else:
                X_train = jag_inp[tr_id,:]
                y_sca_train = jag_sca[tr_id,:]
            y_img_train = jag_img[tr_id,:]

            # Unlike 'fft-scattering-coef', the iteration per subset of data is only dependent on 'split_n'. With 'it_max//split_n',
            # every subset of data will be given equal (statistically) no. of chance to be trained on.
            for it in range(it_max//split_n):

                if X_train.shape[0] < batch_size:
                    batch_size = X_train.shape[0]

                # The training indexes per subset of data is randomised
                randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
                x_mb = X_train[randid,:]
                y_img_mb = y_img_train[randid,:]

                y_sca_mb = y_sca_train[randid,:]

                # Feed dictionary for training (or the inputs to the training model)
                fd = {x: x_mb, y_sca: y_sca_mb,y_img:y_img_mb,train_mode:True}

                for _ in range(10):
                    _,dloss = sess.run([JagNet_MM.D_solver,JagNet_MM.loss_disc],feed_dict=fd)

                gloss0,gloss1,gadv = sess.run([JagNet_MM.loss_gen0,
                                                JagNet_MM.loss_gen1,
                                                JagNet_MM.loss_adv],
                                                feed_dict=fd)

                for _ in range(1):
                    _ = sess.run([JagNet_MM.G0_solver],feed_dict=fd)

                if (it_total+1) % 100 == 0:
                    print('Fidelity -- Iter: {}; Forward: {:.4f}; Inverse: {:.4f}'
                        .format(it_total+1, gloss0, gloss1))
                    if (it_total+1) % 500 == 0:
                        print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}'.format(dloss,gadv))
                    else:
                        print('Adversarial -- Disc: {:.4f}; Gen: {:.4f}\n'.format(dloss,gadv))


                if (it_total+1) % 500 == 0:

                    nTest=16
                    x_test_mb = X_test[-nTest:,:]

                    summary_val = sess.run(merged,feed_dict={x:X_test,y_sca:y_sca_test,y_img:y_img_test,train_mode:False})

                    writer.add_summary(summary_val, it_total+1)

                    samples,samples_x = sess.run([y_img_out,JagNet_MM.input_cyc],
                                                feed_dict={x: x_test_mb,train_mode:False})
                    data_dict= {}
                    data_dict['samples'] = samples
                    data_dict['samples_x'] = samples_x
                    data_dict['y_sca'] = y_sca_test
                    data_dict['y_img'] = y_img_test
                    data_dict['x'] = x_test_mb

                    # Saves testing results in './surrogate_outs'.
                    test_imgs_plot(fdir,it_total+1,data_dict, complex_mode, dataset)

                # Saves trained model in './surrogate_model_weights'.
                # Can set which iteration should the model be saved
                if (it_total+1)%(it_max//100)==0:
                    if complex_mode:
                        save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_complex_model_"+str(it_total)+".ckpt")
                    else:
                        save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_model_"+str(it_total)+".ckpt")

                it_total = it_total + 1
                
    print('Surrogate Training Completed...')
    
    return

if __name__=='__main__':
    # Script starts to enter here
    run()

### srun -n 1 -N 1 -p pbatch -A lbpm --time=3:00:00 --pty /bin/sh
