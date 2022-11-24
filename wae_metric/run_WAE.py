import os
import shutil
import pickle as pkl
import argparse

import tensorflow as tf

import numpy as np
np.random.seed(4321)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler

from .model_AVB import *

LATENT_SPACE_DIM = 30           # try 80 for dataset == 'fft-scattering-coef'


def sample_z(L,dim,type='uniform'):
    if type == 'hypercircle':
        theta=[w/np.sqrt((w**2).sum()) for w in np.random.normal(size=(L,dim))]
    elif type == 'uniform':
        theta = 5*(1-2*np.random.rand(L, dim))
    elif type == 'normal':
        theta = np.random.randn(L,dim)

    return np.asarray(theta)

def load_dataset(complex_mode):

    datapath = './data/icf-jag-10k/'

    if complex_mode:
        jag_img = np.load(datapath+'jag10K_images_complex.npy')
    else:
        jag_img = np.load(datapath+'jag10K_images.npy')
    jag_sca = np.load(datapath+'jag10K_0_scalars.npy')
    jag_inp = np.load(datapath+'jag10K_params.npy')

    return jag_inp,jag_sca,jag_img

def load_fft_dataset_actual(fft_npy_datapath, files_list_split):

    first_data_flag = 1

    for npy in files_list_split:
        if first_data_flag == 1:
            fft_data = np.load(fft_npy_datapath + npy)
            first_data_flag = 0
        else:
            fft_data_cur = np.load(fft_npy_datapath + npy)
            fft_data = np.vstack([fft_data, fft_data_cur])

    return fft_data

def run(**kwargs):

    fdir = kwargs.get('fdir','./wae_metric/ae_outs')
    modeldir = kwargs.get('modeldir','./wae_metric/ae_model_weights')
    dataset = kwargs.get('dataset')
    vizdir = kwargs.get('vizdir','graphs')
    lam = kwargs.get('lam',1e-4)
    dim_z = kwargs.get('dimz',LATENT_SPACE_DIM)
    complex_mode = kwargs.get('complex_mode')
    split_n = kwargs.get('split_n')
    num_npy = kwargs.get('num_npy')
    batch_size = kwargs.get('batch_size')

    if dataset == 'fft-scattering-coef':
        complex_mode = True
        print('fft-scattering-coef dataset is selected...')
    else:
        print('icf-jag dataset is selected...')

    if complex_mode:
        print('Complex Mode is selected...')
    else:
        print('Non-complex Mode is selected...')

    if not os.path.exists(fdir):
        os.makedirs(fdir)

    if not os.path.exists(modeldir):
        os.makedirs(modeldir)

    print('Loading dataset...')

    if dataset == 'fft-scattering-coef':

        fft_datapath = './data/fft-scattering-coef-40k/'

        fft_img_datapath = fft_datapath + 'fft40K_images/'
        fft_img_files_list = os.listdir(fft_img_datapath)

        # fft_inp_datapath = fft_datapath + 'fft40K_params/'
        # fft_inp_files_list = os.listdir(fft_inp_datapath)

        fft_data_size_per_npy_set = num_npy * 100                   # Each npy has 100 data, thus total data is num_npy * 100

        tr_id = np.random.choice(fft_data_size_per_npy_set,int(fft_data_size_per_npy_set*0.95),replace=False)
        te_id = list(set(range(fft_data_size_per_npy_set)) - set(tr_id))

        # X_test_set = fft_img[te_id,:]           # same variable for y_test_set
        # # y_test_set = fft_img[te_id,:]

        dim_image = 64*64*16

    else:
        jag_inp, jag_sca, jag_img = load_dataset(complex_mode)
        jag = np.hstack((jag_img,jag_sca))

        tr_id = np.random.choice(jag.shape[0],int(jag.shape[0]*0.95),replace=False)
        te_id = list(set(range(jag.shape[0])) - set(tr_id))

        X_img_test_set = jag_img[te_id,:]
        X_sca_test_set = jag_sca[te_id,:]
        X_test_set = jag[te_id,:]           # same variable for y_test_set
        # y_test_set = jag[te_id,:]

        dim_image = jag.shape[1]
    
    print('No. of training dataset: ', len(tr_id))
    print('No. of testing dataset: ', len(te_id))

    # X_train  = jag[tr_id,:]             # same variable for y_train
    # # y_train = jag[tr_id,:]

    print('Splitting dataset indexes...')

    sub_X_train_len = len(tr_id)/split_n

    if sub_X_train_len % 1 == 0:
        sub_X_train_len = int(sub_X_train_len)
        tr_id_split = [tr_id[i*sub_X_train_len:i*sub_X_train_len+sub_X_train_len] for i in range(split_n)]
    else:
        sub_X_train_len = len(tr_id)//split_n
        tr_id_split = [tr_id[i*sub_X_train_len:i*sub_X_train_len+sub_X_train_len] if i < sub_X_train_len else tr_id[i*sub_X_train_len:] for i in range(split_n+1)]

    if batch_size > sub_X_train_len:
        batch_size = sub_X_train_len
        print('Selected batch size is larger than splitted batch size... Using splitted batch size instead...')

    print('Batch Size: ', batch_size)

    print('Initialising model...')

    # Image  to Image
    if complex_mode:
        y = tf.placeholder(tf.complex64, shape=[None, dim_image])
        z = tf.placeholder(tf.complex64, shape=[None, dim_z])
    else:
        y = tf.placeholder(tf.float32, shape=[None, dim_image])
        z = tf.placeholder(tf.float32, shape=[None, dim_z])
    train_mode = tf.placeholder(tf.bool,name='train_mode')

    # Model initialisation

    # Multi-modal Wasserstein Autoencoder (see architecture in paper)

    z_sample = gen_encoder_FCN(y, dim_z,train_mode, False, complex_mode)

    y_recon = var_decoder_FCN(z_sample, dim_image,train_mode, False, complex_mode)

    # AE Discriminator (architecture not in paper)

    D_sample = discriminator_FCN(y_recon, z_sample, None, complex_mode)
    D_prior = discriminator_FCN(y, z, True, complex_mode)

    # Model initialisation ends

    if complex_mode:
        img_loss = tf.reduce_mean(tf.square(tf.abs(y_recon-y)))
        rec_error = tf.reduce_mean(tf.nn.l2_loss(tf.abs(y_recon-y)))
    else:
        img_loss = tf.reduce_mean(tf.square(y_recon-y))
        rec_error = tf.reduce_mean(tf.nn.l2_loss(y_recon-y))
    D_loss,gen_error = compute_adv_loss(D_prior,D_sample, complex_mode)
    G_loss = rec_error+lam*gen_error

    t_vars = tf.global_variables()
    d_vars = [var for var in t_vars if 'discriminator' in var.name]
    g_vars = list(set(t_vars)-set(d_vars))

    D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss,var_list=d_vars)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(G_loss,var_list=g_vars)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    rec_summary = tf.summary.scalar(name='Recon_MSE', tensor=img_loss)
    disc_summary = tf.summary.scalar(name='Disc_Loss',tensor=D_loss)
    gen_summary = tf.summary.scalar(name='Gen_Loss',tensor=gen_error)
    merged = tf.summary.merge_all()

    writer = tf.summary.FileWriter('./train_graphs/{}'.format(vizdir), sess.graph)
    writer_test = tf.summary.FileWriter('./test_graphs/{}'.format(vizdir), sess.graph)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(modeldir)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("************ Model Restored! **************")
    else:
        print("******* No Pretrained Model Found! ********")

    print('Training starts...')
        
    it_total = 0
    if dataset == 'fft-scattering-coef':
        it_max = 400000
    else:
        it_max = 100000

    print('A total of ' + str(it_max) + ' data iteration is selected for the training...')

    if dataset == 'fft-scattering-coef':

        for i in range(num_npy):
            fft_img_files_list_split = fft_img_files_list[i*num_npy:i*num_npy+num_npy]
            # fft_inp_files_list_split = fft_inp_files_list[i*num_npy:i*num_npy+num_npy]

            fft_img = load_fft_dataset_actual(fft_img_datapath, fft_img_files_list_split)
            # fft_inp = load_fft_dataset_actual(fft_inp_datapath, fft_inp_files_list_split)

            for tr_id in tr_id_split:
                X_train  = fft_img[tr_id,:]
                X_test_set = fft_img[te_id,:]

                for it in range(0, (it_max//num_npy)//split_n):
                    if X_train.shape[0] < batch_size:
                        batch_size = X_train.shape[0]
                    randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
                    y_mb = X_train[randid,:]
                    # X_mb = X_train[randid,:]

                    z_mb = sample_z(batch_size,dim_z)

                    fd = {y:y_mb, train_mode: True,z:z_mb}
                    for i in range(1):
                        _, G_loss_curr,tmp1,tmp2 = sess.run([G_solver, G_loss,rec_error,gen_error],
                                                feed_dict=fd)
                    for i in range(1):
                        _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict=fd)

                    if (it_total+1) % 100 ==0:
                        summary = sess.run(merged,feed_dict=fd)
                        writer.add_summary(summary, it_total)

                    if (it_total+1) % 1000 == 0:
                        print('Iter: {}; Recon_Error = : {:.4}, G_Loss: {:.4}; D_Loss: {:.4}'
                            .format(it_total+1, tmp1, tmp2, D_loss_curr))

                        z_test = sample_z(len(te_id),dim_z)
                        fd = {train_mode:False, y:X_test_set, z:z_test}
                        samples,summary_val = sess.run([y_recon,merged],feed_dict=fd)

                        writer_test.add_summary(summary_val, it_total)

                        data_dict= {}
                        data_dict['samples'] = samples
                        
                        data_dict['y_img'] = X_test_set

                        ae_test_imgs_plot(fdir,it_total,data_dict, complex_mode, dataset)

                    if (it_total+1)%(it_max//10)==0:
                        if complex_mode:
                            save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_complex_model_"+str(it_total)+".ckpt")
                        else:
                            save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_model_"+str(it_total)+".ckpt")

                    it_total = it_total + 1

    else:
        for tr_id in tr_id_split:
            X_train  = jag[tr_id,:]
            for it in range(0, it_max//split_n):
                if X_train.shape[0] < batch_size:
                    batch_size = X_train.shape[0]
                randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
                y_mb = X_train[randid,:]
                # X_mb = X_train[randid,:]

                z_mb = sample_z(batch_size,dim_z)

                fd = {y:y_mb, train_mode: True,z:z_mb}
                for i in range(1):
                    _, G_loss_curr,tmp1,tmp2 = sess.run([G_solver, G_loss,rec_error,gen_error],
                                            feed_dict=fd)
                for i in range(1):
                    _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict=fd)

                if (it_total+1) % 100 ==0:
                    summary = sess.run(merged,feed_dict=fd)
                    writer.add_summary(summary, it_total)

                if (it_total+1) % 1000 == 0:
                    print('Iter: {}; Recon_Error = : {:.4}, G_Loss: {:.4}; D_Loss: {:.4}'
                        .format(it_total+1, tmp1, tmp2, D_loss_curr))

                    z_test = sample_z(len(te_id),dim_z)
                    fd = {train_mode:False, y:X_test_set, z:z_test}
                    samples,summary_val = sess.run([y_recon,merged],feed_dict=fd)

                    writer_test.add_summary(summary_val, it_total)

                    data_dict= {}
                    data_dict['samples'] = samples
                    data_dict['y_sca'] = X_sca_test_set
                    data_dict['y_img'] = X_img_test_set

                    ae_test_imgs_plot(fdir,it_total,data_dict, complex_mode, dataset)

                if (it_total+1)%(it_max//10)==0:
                    if complex_mode:
                        save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_complex_model_"+str(it_total)+".ckpt")
                    else:
                        save_path = saver.save(sess, "./"+modeldir+"/" + dataset + "_model_"+str(it_total)+".ckpt")

                it_total = it_total + 1

    # else:
    #     for it in range(0,100000):
    #         randid = np.random.choice(X_train.shape[0],batch_size,replace=False)
    #         y_mb = X_train[randid,:]
    #         X_mb = X_train[randid,:]


    #         z_mb = sample_z(batch_size,dim_z)

    #         fd = {y:y_mb, train_mode: True,z:z_mb}
    #         for i in range(1):
    #             _, G_loss_curr,tmp1,tmp2 = sess.run([G_solver, G_loss,rec_error,gen_error],
    #                                     feed_dict=fd)
    #         for i in range(1):
    #             _, D_loss_curr = sess.run([D_solver, D_loss],feed_dict=fd)

    #         if (it+1) % 100 ==0:
    #             summary = sess.run(merged,feed_dict=fd)
    #             writer.add_summary(summary, it)

    #         if (it+1) % 1000 == 0:
    #             print('Iter: {}; Recon_Error = : {:.4}, G_Loss: {:.4}; D_Loss: {:.4}'
    #                 .format(it+1, tmp1, tmp2, D_loss_curr))

    #             z_test = sample_z(len(te_id),dim_z)
    #             fd = {train_mode:False, y:X_test_set, z:z_test}
    #             samples,summary_val = sess.run([y_recon,merged],feed_dict=fd)

    #             writer_test.add_summary(summary_val, it)

    #         if (it+1)%1000==0:
    #             save_path = saver.save(sess, "./"+modeldir+"/model_"+str(it)+".ckpt")

    print('Autoencoder Training Completed...')

    return

if __name__=='__main__':
    run(datapath='../data/',vizdir='viz',modeldir='pretrained_model',lam=1e-2,dimz=LATENT_SPACE_DIM)
