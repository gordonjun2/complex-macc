import os
import shutil
import pickle as pkl
import argparse

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler, scale

import warnings

IMAGE_SIZE = 50
warnings.filterwarnings('ignore')


def log(x):
    return tf.log(x + 1e-8)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_eps(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def GANloss(D_real,D_fake, complex_mode):
    if complex_mode:
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.abs(D_real), labels=tf.ones_like(tf.abs(D_real))))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.abs(D_fake), labels=tf.zeros_like(tf.abs(D_fake))))
        D_loss = D_loss_real + D_loss_fake
        G_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.abs(D_fake), labels=tf.ones_like(tf.abs(D_fake))))
    else:
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
    return D_loss,G_adv

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def weight_variable_xavier_initialized(shape, constant=1, name=None):
    stddev = constant * np.sqrt(2.0 / (shape[2] + shape[3]))
    return weight_variable(shape, stddev=stddev, name=name)

def weight_selu(shape, transpose=False,name=None):
    if transpose:
        stddev = np.sqrt(1.0 / (shape[0] * shape[1]*shape[2]))
    else:
        stddev = np.sqrt(1.0 / (shape[0] * shape[1]*shape[3]))

    return weight_variable(shape, stddev=stddev, name=name)


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.random_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_transpose_strided(x, W, b, output_shape=None):
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

def conv2d(x, W,filter_size=5,strides=(1,2,2,1)):
  return tf.nn.conv2d(x, W, strides=strides, padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def conv1d(x,W,strides=1,act='relu'):
    conv = tf.nn.conv1d(x,W,stride=strides,padding='SAME')
    if act=='selu':
        h = selu(conv)
    else:
        h = tf.nn.relu(conv)

    return h

def bn(x,is_training,name):
    return batch_norm(x, decay=0.9, center=True, scale=True,updates_collections=None,is_training=is_training,
    reuse=None,
    trainable=True,
    scope=name)

#     if is_training:
#             return batch_norm(x, decay=0.9, center=True, scale=True,updates_collections=None,is_training=is_training,
#             reuse=None, 
#             trainable=True, 
#             scope=name)
        
#     else:
#         # is_training = True (forced), trainable = False
#         # If is_training = False instead, there will be NaN if complex_mode is being used. Not sure why ...
#         # Note: Normalization is performed using the mean and variance of the minibatch for Tensorflow 1.* version (to be used).
#         #       For Tensorflow 2.* version, sliding mean and sliding variance is used instead.
#         #       (https://blog.csdn.net/Strive_For_Future/article/details/115243512)
#         #       Either way, as long as trainable = False, batch normalization layer is freezed.
#         #       However, the original intention of the authors is to use sliding mean and sliding variance ...
#         return batch_norm(x, decay=0.9, center=True, scale=True,updates_collections=None,is_training=True,
#         reuse=None,
#         trainable=False,
#         scope=name)

def batch_norm_custom(x, n_out, phase_train, scope='bn', decay=0.99, eps=1e-5,reuse=None):

    with tf.variable_scope(scope) as scope:
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def selu(x,name="selu"):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
    return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def filter_output(samples):
    s = []
    for _label in samples:
        mask = np.where(_label<=np.median(_label),0,1)
        _label = mask*_label
        mask2 = np.where(_label>np.percentile(_label,99),1,0)
        p = np.percentile(_label,99)
        _label[np.where(mask2)] = p
        s.append(_label)
    return s

def separable_conv2d_util(X,filter_sizes,names,depth_multiplier=1,strides=[1,1,1,1]):
    zz = filter_sizes[:2]
    in_channels = filter_sizes[2]
    out_channels = filter_sizes[3]
    dm = depth_multiplier
    wconv1_dw = weight_variable_xavier_initialized([zz[0],zz[1],in_channels,dm],name=names[0])
    wconv1_pw = weight_variable_xavier_initialized([1,1,in_channels*dm,out_channels],name=names[1])
    b_conv1 = bias_variable([out_channels],name=names[2])
    h = tf.nn.separable_conv2d(X,wconv1_dw,wconv1_pw,strides,padding='SAME') + b_conv1
    return h

def concat(tensors, axis, *args, **kwargs):
       return tf.concat(tensors, axis, *args, **kwargs)

def conv_cond_concat(x, y):
   """Concatenate conditioning vector on feature map axis."""
   x_shapes = x.get_shape()
   y_shapes = y.get_shape()
   return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def plot_save(samples,figsize=(4,4),wspace=0.05,hspace=0.05,cmap='gray',resize=False):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(figsize[0], figsize[1])
    gs.update(wspace=wspace, hspace=hspace)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if(resize):
            sample = imresize(sample,(128,128))

        plt.imshow(sample, cmap=cmap)
        title="{:.2f}".format(np.linalg.norm(sample))
        # ax.text(16, 16, title,fontsize=8,
               # bbox={'facecolor':'white', 'alpha':0.9, 'pad':1})
    return fig


def plot_line(dataset, samples,gt,bar=True):
    # plt.rcParams["figure.figsize"] = (10,20)
    if bar :
        fig = plt.figure(figsize=(20, 20))
    else:
        fig = plt.figure(figsize=(20, 20))

    #
    width=0.35

    if dataset == 'fft-scattering-coef':
        params = ['RJ_scaled', 'phiJ_scaled', 'squintJ_scaled',
                  'vJ_scaled', 'fc_scaled', 'prf_scaled',
                  'x_px_scaled', 'y_px_scaled', 'x_off_scaled',
                  'y_off_scaled']

    else:
        params = ['stopping_mult', 'radiation_mult', 'ablation_cv', 'Vi',
                'conduction_mult', 'shape_model_initial_velocity_amplitude',
                'shape_model_initial_velocities:(1, 0)',
                    'shape_model_initial_velocities:(1, 1)',
                    'shape_model_initial_velocities:(2, 0)',
                    'shape_model_initial_velocities:(2, 1)',
                    'shape_model_initial_velocities:(2, 2)']
    ind = np.arange(len(params))

    for i, sample in enumerate(samples):
        ax = plt.subplot(10,10,i+1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if bar:
            if gt is not None:
                ax.barh(ind,gt[i,:],width,color='red')
            else:
                ax.barh(ind,width,color='red')
            ax.barh(ind+width,sample,width, color='blue')
            # plt.yticks(ind ,params,rotation=30)
            # ax.legend(['GT', 'Pred'])
        else:
            if gt is not None:
                plt.plot(gt[i,:])
            plt.plot(sample,'r.-')

    return fig

def plot(samples,immax=None,immin=None):
    # plt.rcParams["figure.figsize"] = (10,10)
    IMAGE_SIZE = 64
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if immax is not None:
            plt.imshow(sample.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='winter',vmax=immax[i],vmin=immin[i])
    return fig

def meanEuclidDist(gt, pred):

    dist = np.linalg.norm(gt - pred, axis=1)

    mean_dist = np.mean(dist)

    return mean_dist

def test_imgs_plot(fdir,batch,data_dict, complex_mode, dataset):
    i = batch
    samples = data_dict['samples']
    samples_x = data_dict['samples_x']
    if dataset != 'fft-scattering-coef':
        y_sca_test = data_dict['y_sca']
    y_img_test = data_dict['y_img']
    x_test_mb = data_dict['x']

    nTest = x_test_mb.shape[0]
    # nTest = 100
    
    if dataset != 'fft-scattering-coef':
        y_sca_test_mb = y_sca_test[-nTest:,:]

    if dataset == 'fft-scattering-coef':

        mean_dist = meanEuclidDist(samples[-nTest:,:], y_img_test[-nTest:,:])
        print('Mean Euclidean Distance between Ground Truth images and Predicted images: {:.4f}\n'
            .format(mean_dist))

        idx = np.random.choice(range(16),1)
        y_img_test_ = y_img_test[-nTest:,:]
        y_img_test_mb = y_img_test_.reshape(-1,64,64,16)[:,:,:,idx].reshape(-1,4096)
        samples_y_img = samples[-nTest:,:].reshape(-1,64,64,16)
    else:

        mean_dist = meanEuclidDist(samples[-nTest:,:16384], y_img_test[-nTest:,:16384])
        print('Mean Euclidean Distance between Ground Truth images and Predicted images: {:.4f}\n'
            .format(mean_dist))

        idx = np.random.choice(range(4),1)
        y_img_test_ = y_img_test[-nTest:,:16384]
        y_img_test_mb = y_img_test_.reshape(-1,64,64,4)[:,:,:,idx].reshape(-1,4096)
        samples_y_sca = samples[-nTest:,16384:]
        samples_y_img = samples[-nTest:,:16384].reshape(-1,64,64,4)
    samples_y_img_plot = samples_y_img[:,:,:,idx]

    if dataset != 'fft-scattering-coef':
        fig = plot_line(dataset, samples_y_sca,y_sca_test_mb,bar=False)
        plt.savefig('{}/y_sca_{}.png'
                    .format(fdir,str(i).zfill(3)), bbox_inches='tight')
        plt.close()

    if complex_mode:
        # Real
        fig = plot(np.real(samples_y_img_plot),immax=np.max(np.real(y_img_test_mb),axis=1),immin=np.min(np.real(y_img_test_mb),axis=1))
        plt.savefig('{}/y_real_img_{}_{}.png'
                    .format(fdir,str(i).zfill(3),str(idx)), bbox_inches='tight')
        plt.close()

        # Imaginary
        fig = plot(np.imag(samples_y_img_plot),immax=np.max(np.imag(y_img_test_mb),axis=1),immin=np.min(np.imag(y_img_test_mb),axis=1))
        plt.savefig('{}/y_imag_img_{}_{}.png'
                    .format(fdir,str(i).zfill(3),str(idx)), bbox_inches='tight')
        plt.close()

    else:
        fig = plot(samples_y_img_plot,immax=np.max(y_img_test_mb,axis=1),immin=np.min(y_img_test_mb,axis=1))
        plt.savefig('{}/y_img_{}_{}.png'
                    .format(fdir,str(i).zfill(3),str(idx)), bbox_inches='tight')
        plt.close()

    fig = plot_line(dataset, samples_x,x_test_mb,bar=False)
    plt.savefig('{}/x_{}.png'
                .format(fdir,str(i).zfill(3)), bbox_inches='tight')
    plt.close()
    return

def inference_results(infer_dir, gt_x, pred_x, pred_y, complex_mode, dataset):

    if dataset == 'fft-scattering-coef':
        pred_y_img = pred_y.reshape(-1,64,64,16)
    else:

        pred_y_sca =  pred_y[:,16384:]
        pred_y_img =  pred_y[:,:16384].reshape(-1,64,64,4)

    pred_y_img_plot = pred_y_img

    if dataset != 'fft-scattering-coef':
        fig = plot_line(dataset, pred_y_sca,None,bar=False)
        plt.savefig('{}/y_sca_inferred.png'
                    .format(infer_dir), bbox_inches='tight')
        plt.close()

        print('Predicted y_sca (output parameters): ', pred_y_sca)

    num_channels = np.shape(pred_y_img_plot)[3]

    if complex_mode:

        for idx in range(num_channels):

            pred_y_img_test = pred_y_img[:,:,:,idx].reshape(-1,4096)

            # Real
            fig = plot(np.real(pred_y_img_plot[:, :, :, idx]),immax=np.max(np.real(pred_y_img_test),axis=1),immin=np.min(np.real(pred_y_img_test),axis=1))
            plt.savefig('{}/y_real_img_inferred_{}.png'
                        .format(infer_dir,str(idx)), bbox_inches='tight')
            plt.close()

            # Imaginary
            fig = plot(np.imag(pred_y_img_plot[:, :, :, idx]),immax=np.max(np.imag(pred_y_img_test),axis=1),immin=np.min(np.imag(pred_y_img_test),axis=1))
            plt.savefig('{}/y_imag_img_inferred_{}.png'
                        .format(infer_dir,str(idx)), bbox_inches='tight')
            plt.close()

    else:

        for idx in range(num_channels):

            pred_y_img_test = pred_y_img[:,:,:,idx].reshape(-1,4096)
            
            fig = plot(pred_y_img_plot[:, :, :, idx],immax=np.max(pred_y_img_test,axis=1),immin=np.min(pred_y_img_test,axis=1))
            plt.savefig('{}/y_img_inferred_{}.png'
                        .format(infer_dir,str(idx)), bbox_inches='tight')
            plt.close()


    fig = plot_line(dataset, pred_x, gt_x, bar=False)
    plt.savefig('{}/x_inferred.png'
                .format(infer_dir), bbox_inches='tight')
    plt.close()

    print('Predicted x (input parameters): ', pred_x)

    return
