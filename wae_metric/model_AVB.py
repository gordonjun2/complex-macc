'''Models'''

import tensorflow as tf

from .utils import *
from .utils import spectral_norm as SN


def xavier_init(size,name=None):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev,name=name)

'''FCN Encoder'''
def gen_encoder_FCN(x, n_output, train_mode, reuse=False, complex_mode=False):
    n_hidden = [32,256,128]
    # n_hidden = [1024,256,32]

    with tf.variable_scope("wae_encoder",reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        if complex_mode:

            x_real = tf.real(x)
            x_imag = tf.imag(x)
            
            w0_real = tf.get_variable('w0_real', [x.get_shape()[1], n_hidden[0]], initializer=w_init)
            b0_real = tf.get_variable('b0_real', [n_hidden[0]], initializer=b_init)
            tf_real_real = tf.matmul(x_real, w0_real) + b0_real
            tf_imag_real = tf.matmul(x_imag, w0_real) + b0_real

            w0_imag = tf.get_variable('w0_imag', [x.get_shape()[1], n_hidden[0]], initializer=w_init)
            b0_imag = tf.get_variable('b0_imag', [n_hidden[0]], initializer=b_init)
            tf_real_imag = tf.matmul(x_real, w0_imag) + b0_imag
            tf_imag_imag = tf.matmul(x_imag, w0_imag) + b0_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag

            h0_real = bn(real_out, train_mode,"bn1_real")
            h0_imag = bn(imag_out, train_mode,"bn1_imag")
            
#             # BN issue
#             h0_real = tf.Print(h0_real, [h0_real])
#             h0_imag = tf.Print(h0_imag, [h0_imag])
#             #
            
            h0_real = tf.nn.elu(h0_real)
            h0_imag = tf.nn.elu(h0_imag)

            h0 = tf.complex(h0_real, h0_imag)

        else:
            # inputs = tf.concat(axis=1, values=[x, eps])
            inputs = x
            # w0 = tf.get_variable('w0', [x.get_shape()[1] + eps.get_shape()[1], n_hidden[0]], initializer=w_init)
            w0 = tf.get_variable('w0', [x.get_shape()[1], n_hidden[0]], initializer=w_init)
            b0 = tf.get_variable('b0', [n_hidden[0]], initializer=b_init)
            h0 = tf.matmul(inputs, w0) + b0
            h0 = bn(h0, train_mode,"bn1")
            h0 = tf.nn.elu(h0)
        
        # 2nd hidden layer
        if complex_mode:
            
            h0_real = tf.real(h0)
            h0_imag = tf.imag(h0)

            w1_real = tf.get_variable('w1_real', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1_real = tf.get_variable('b1_real', [n_hidden[1]], initializer=b_init)
            tf_real_real = tf.matmul(h0_real, w1_real) + b1_real
            tf_imag_real = tf.matmul(h0_imag, w1_real) + b1_real

            w1_imag = tf.get_variable('w1_imag', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1_imag = tf.get_variable('b1_imag', [n_hidden[1]], initializer=b_init)
            tf_real_imag = tf.matmul(h0_real, w1_imag) + b1_imag
            tf_imag_imag = tf.matmul(h0_imag, w1_imag) + b1_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag

            h1_real = bn(real_out, train_mode,"bn2_real")
            h1_imag = bn(imag_out, train_mode,"bn2_imag")

            h1_real = tf.nn.tanh(h1_real)
            h1_imag = tf.nn.tanh(h1_imag)

            h1 = tf.complex(h1_real, h1_imag)
        else:
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden[1]], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            h1 = bn(h1, train_mode,"bn2")
            h1 = tf.nn.tanh(h1)

        # 3rd hidden layer
        if complex_mode:

            h1_real = tf.real(h1)
            h1_imag = tf.imag(h1)

            w2_real = tf.get_variable('w2_real', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2_real = tf.get_variable('b2_real', [n_hidden[2]], initializer=b_init)
            tf_real_real = tf.matmul(h1_real, w2_real) + b2_real
            tf_imag_real = tf.matmul(h1_imag, w2_real) + b2_real

            w2_imag = tf.get_variable('w2_imag', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2_imag = tf.get_variable('b2_imag', [n_hidden[2]], initializer=b_init)
            tf_real_imag = tf.matmul(h1_real, w2_imag) + b2_imag
            tf_imag_imag = tf.matmul(h1_imag, w2_imag) + b2_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag

            h2_real = bn(real_out, train_mode,"bn3_real")
            h2_imag = bn(imag_out, train_mode,"bn3_imag")

            h2_real = tf.nn.tanh(h2_real)
            h2_imag = tf.nn.tanh(h2_imag)

            h2 = tf.complex(h2_real, h2_imag)

        else:
            w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2 = tf.get_variable('b2', [n_hidden[2]], initializer=b_init)
            h2 = tf.matmul(h1, w2) + b2
            h2 = bn(h2, train_mode,"bn3")
            h2 = tf.nn.tanh(h2)

        # output layer
        if complex_mode:

            h2_real = tf.real(h2)
            h2_imag = tf.imag(h2)

            wout_real = tf.get_variable('wout_real', [h2.get_shape()[1], n_output], initializer=w_init)
            bout_real = tf.get_variable('bout_real', [n_output], initializer=b_init)
            tf_real_real = tf.matmul(h2_real, wout_real) + bout_real
            tf_imag_real = tf.matmul(h2_imag, wout_real) + bout_real

            wout_imag = tf.get_variable('wout_imag', [h2.get_shape()[1], n_output], initializer=w_init)
            bout_imag = tf.get_variable('bout_imag', [n_output], initializer=b_init)
            tf_real_imag = tf.matmul(h2_real, wout_imag) + bout_imag
            tf_imag_imag = tf.matmul(h2_imag, wout_imag) + bout_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            
#             real_out = tf.Print(real_out, [real_out])
#             imag_out = tf.Print(imag_out, [imag_out])

            z = tf.complex(real_out, imag_out)

        else:
            wout = tf.get_variable('wout', [h2.get_shape()[1], n_output], initializer=w_init)
            bout = tf.get_variable('bout', [n_output], initializer=b_init)
            z = tf.matmul(h2, wout) + bout

    return z


def var_decoder_FCN(z, n_output, train_mode, reuse=False, complex_mode=False):
    n_hidden = [64,128,256]
    # n_hidden = [32,256,1024]
    with tf.variable_scope("wae_decoder", reuse=reuse):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        # 1st hidden layer
        if complex_mode:

            z_real = tf.real(z)
            z_imag = tf.imag(z)

            dw0_real = tf.get_variable('dw0_real', [z.get_shape()[1], n_hidden[0]], initializer=w_init)
            db0_real = tf.get_variable('db0_real', [n_hidden[0]], initializer=b_init)
            tf_real_real = tf.matmul(z_real, dw0_real) + db0_real
            tf_imag_real = tf.matmul(z_imag, dw0_real) + db0_real

            dw0_imag = tf.get_variable('dw0_imag', [z.get_shape()[1], n_hidden[0]], initializer=w_init)
            db0_imag = tf.get_variable('db0_imag', [n_hidden[0]], initializer=b_init)
            tf_real_imag = tf.matmul(z_real, dw0_imag) + db0_imag
            tf_imag_imag = tf.matmul(z_imag, dw0_imag) + db0_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h0 = tf.complex(real_out, imag_out)

            # h0 = bn(h0, train_mode,"bn0")

            h0_real = tf.real(h0)
            h0_imag = tf.imag(h0)

            h0_real = tf.nn.elu(h0_real)
            h0_imag = tf.nn.elu(h0_imag)

            h0 = tf.complex(h0_real, h0_imag)

        else:
            dw0 = tf.get_variable('dw0', [z.get_shape()[1], n_hidden[0]], initializer=w_init)
            db0 = tf.get_variable('db0', [n_hidden[0]], initializer=b_init)
            h0 = tf.matmul(z, dw0) + db0
            # h0 = bn(h0,train_mode,"bn0")
            h0 = tf.nn.elu(h0)

        # 2nd hidden layer
        if complex_mode:

            h0_real = tf.real(h0)
            h0_imag = tf.imag(h0)

            w1_real = tf.get_variable('w1_real', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1_real = tf.get_variable('b1_real', [n_hidden[1]], initializer=b_init)
            tf_real_real = tf.matmul(h0_real, w1_real) + b1_real
            tf_imag_real = tf.matmul(h0_imag, w1_real) + b1_real

            w1_imag = tf.get_variable('w1_imag', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1_imag = tf.get_variable('b1_imag', [n_hidden[1]], initializer=b_init)
            tf_real_imag = tf.matmul(h0_real, w1_imag) + b1_imag
            tf_imag_imag = tf.matmul(h0_imag, w1_imag) + b1_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h1 = tf.complex(real_out, imag_out)

            # h1 = bn(h1, train_mode,"bn1")

            h1_real = tf.real(h1)
            h1_imag = tf.imag(h1)

            h1_real = tf.nn.tanh(h1_real)
            h1_imag = tf.nn.tanh(h1_imag)

            h1 = tf.complex(h1_real, h1_imag)

        else:
            w1 = tf.get_variable('w1', [h0.get_shape()[1], n_hidden[1]], initializer=w_init)
            b1 = tf.get_variable('b1', [n_hidden[1]], initializer=b_init)
            h1 = tf.matmul(h0, w1) + b1
            # h1 = bn(h1,train_mode,"bn1")
            h1 = tf.nn.tanh(h1)

        # 3rd hidden layer
        if complex_mode:

            h1_real = tf.real(h1)
            h1_imag = tf.imag(h1)

            w2_real = tf.get_variable('w2_real', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2_real = tf.get_variable('b2_real', [n_hidden[2]], initializer=b_init)
            tf_real_real = tf.matmul(h1_real, w2_real) + b2_real
            tf_imag_real = tf.matmul(h1_imag, w2_real) + b2_real

            w2_imag = tf.get_variable('w2_imag', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2_imag = tf.get_variable('b2_imag', [n_hidden[2]], initializer=b_init)
            tf_real_imag = tf.matmul(h1_real, w2_imag) + b2_imag
            tf_imag_imag = tf.matmul(h1_imag, w2_imag) + b2_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h2 = tf.complex(real_out, imag_out)

            # h2 = bn(h2,train_mode,"bn2")

            h2_real = tf.real(h2)
            h2_imag = tf.imag(h2)

            h2_real = tf.nn.tanh(h2_real)
            h2_imag = tf.nn.tanh(h2_imag)

            h2 = tf.complex(h2_real, h2_imag)

        else:
            w2 = tf.get_variable('w2', [h1.get_shape()[1], n_hidden[2]], initializer=w_init)
            b2 = tf.get_variable('b2', [n_hidden[2]], initializer=b_init)
            h2 = tf.matmul(h1, w2) + b2
            # h2 = bn(h2,train_mode,"bn2")
            h2 = tf.nn.tanh(h2)

        # output layer
        if complex_mode:

            h2_real = tf.real(h2)
            h2_imag = tf.imag(h2)

            wout_real = tf.get_variable('wout_real', [h2.get_shape()[1], n_output], initializer=w_init)
            bout_real = tf.get_variable('bout_real', [n_output], initializer=b_init)
            tf_real_real = tf.matmul(h2_real, wout_real) + bout_real
            tf_imag_real = tf.matmul(h2_imag, wout_real) + bout_real

            wout_imag = tf.get_variable('wout_imag', [h2.get_shape()[1], n_output], initializer=w_init)
            bout_imag = tf.get_variable('bout_imag', [n_output], initializer=b_init)
            tf_real_imag = tf.matmul(h2_real, wout_imag) + bout_imag
            tf_imag_imag = tf.matmul(h2_imag, wout_imag) + bout_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag

            recon = tf.complex(real_out, imag_out)

        else:
            wout = tf.get_variable('wout', [h2.get_shape()[1], n_output], initializer=w_init)
            bout = tf.get_variable('bout', [n_output], initializer=b_init)
            recon = tf.matmul(h2, wout) + bout

        # print_recon = tf.Print(recon, [recon])

        return recon


def discriminator_FCN(x, z, r=None, complex_mode=False):
    with tf.variable_scope("discriminator", reuse=r):
        w_init = tf.contrib.layers.variance_scaling_initializer()
        b_init = tf.constant_initializer(0.)

        if complex_mode:

            x_real = tf.real(x)
            x_imag = tf.imag(x)
            z_real = tf.real(z)
            z_imag = tf.imag(z)

            inputs_real = tf.concat([x_real, z_real], axis=1)
            inputs_imag = tf.concat([x_imag, z_imag], axis=1)

            D_W1_real = tf.get_variable('D_W1_real', [x.get_shape()[1] + z.get_shape()[1], 512], initializer=w_init)
            D_b1_real = tf.get_variable('D_b1_real', [512], initializer=b_init)
            tf_real_real = tf.matmul(inputs_real, SN(D_W1_real,"sn1_real_real")) + D_b1_real
            tf_imag_real = tf.matmul(inputs_imag, SN(D_W1_real,"sn1_imag_real")) + D_b1_real

            D_W1_imag = tf.get_variable('D_W1_imag', [x.get_shape()[1] + z.get_shape()[1], 512], initializer=w_init)
            D_b1_imag = tf.get_variable('D_b1_imag', [512], initializer=b_init)
            tf_real_imag = tf.matmul(inputs_real, SN(D_W1_imag,"sn1_real_imag")) + D_b1_imag
            tf_imag_imag = tf.matmul(inputs_imag, SN(D_W1_imag,"sn1_imag_imag")) + D_b1_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            
            h1 = tf.complex(real_out, imag_out)

            h1_real = tf.real(h1)
            h1_imag = tf.imag(h1)

            h1_real = tf.nn.leaky_relu(h1_real)
            h1_imag = tf.nn.leaky_relu(h1_imag)

            h1 = tf.complex(h1_real, h1_imag)


            h1_real = tf.real(h1)
            h1_imag = tf.imag(h1)

            D_W2_real = tf.get_variable('D_W2_real', [512, 256], initializer=w_init)
            D_b2_real = tf.get_variable('D_b2_real', [256], initializer=b_init)
            tf_real_real = tf.matmul(h1_real, SN(D_W2_real,"sn2_real_real")) + D_b2_real
            tf_imag_real = tf.matmul(h1_imag, SN(D_W2_real,"sn2_imag_real")) + D_b2_real

            D_W2_imag = tf.get_variable('D_W2_imag', [512, 256], initializer=w_init)
            D_b2_imag = tf.get_variable('D_b2_imag', [256], initializer=b_init)
            tf_real_imag = tf.matmul(h1_real, SN(D_W2_imag,"sn2_real_imag")) + D_b2_imag
            tf_imag_imag = tf.matmul(h1_imag, SN(D_W2_imag,"sn2_imag_imag")) + D_b2_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h2 = tf.complex(real_out, imag_out)

            h2_real = tf.real(h2)
            h2_imag = tf.imag(h2)

            h2_real = tf.nn.leaky_relu(h2_real)
            h2_imag = tf.nn.leaky_relu(h2_imag)

            h2 = tf.complex(h2_real, h2_imag)


            h2_real = tf.real(h2)
            h2_imag = tf.imag(h2)

            D_W3_real = tf.get_variable('D_W3_real', [256, 128], initializer=w_init)
            D_b3_real = tf.get_variable('D_b3_real', [128], initializer=b_init)
            tf_real_real = tf.matmul(h2_real, SN(D_W3_real,"sn3_real_real")) + D_b3_real
            tf_imag_real = tf.matmul(h2_imag, SN(D_W3_real,"sn3_imag_real")) + D_b3_real

            D_W3_imag = tf.get_variable('D_W3_imag', [256, 128], initializer=w_init)
            D_b3_imag = tf.get_variable('D_b3_imag', [128], initializer=b_init)
            tf_real_imag = tf.matmul(h2_real, SN(D_W3_imag,"sn3_real_imag")) + D_b3_imag
            tf_imag_imag = tf.matmul(h2_imag, SN(D_W3_imag,"sn3_imag_imag")) + D_b3_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h3 = tf.complex(real_out, imag_out)

            h3_real = tf.real(h3)
            h3_imag = tf.imag(h3)

            h3_real = tf.nn.leaky_relu(h3_real)
            h3_imag = tf.nn.leaky_relu(h3_imag)

            h3 = tf.complex(h3_real, h3_imag)


            h3_real = tf.real(h3)
            h3_imag = tf.imag(h3)

            D_W4_real = tf.get_variable('D_W4_real', [128, 64], initializer=w_init)
            D_b4_real = tf.get_variable('D_b4_real', [64], initializer=b_init)
            tf_real_real = tf.matmul(h3_real, SN(D_W4_real,"sn4_real_real")) + D_b4_real
            tf_imag_real = tf.matmul(h3_imag, SN(D_W4_real,"sn4_imag_real")) + D_b4_real

            D_W4_imag = tf.get_variable('D_W4_imag', [128, 64], initializer=w_init)
            D_b4_imag = tf.get_variable('D_b4_imag', [64], initializer=b_init)
            tf_real_imag = tf.matmul(h3_real, SN(D_W4_imag,"sn4_real_imag")) + D_b4_imag
            tf_imag_imag = tf.matmul(h3_imag, SN(D_W4_imag,"sn4_imag_imag")) + D_b4_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            h4 = tf.complex(real_out, imag_out)

            h4_real = tf.real(h4)
            h4_imag = tf.imag(h4)

            h4_real = tf.nn.leaky_relu(h4_real)
            h4_imag = tf.nn.leaky_relu(h4_imag)

            h4 = tf.complex(h4_real, h4_imag)


            h4_real = tf.real(h4)
            h4_imag = tf.imag(h4)

            D_W5_real = tf.get_variable('D_W5_real', [64, 1], initializer=w_init)
            D_b5_real = tf.get_variable('D_b5_real', [1], initializer=b_init)
            tf_real_real = tf.matmul(h4_real, SN(D_W5_real,"sn5_real_real")) + D_b5_real
            tf_imag_real = tf.matmul(h4_imag, SN(D_W5_real,"sn5_imag_real")) + D_b5_real

            D_W5_imag = tf.get_variable('D_W5_imag', [64, 1], initializer=w_init)
            D_b5_imag = tf.get_variable('D_b5_imag', [1], initializer=b_init)
            tf_real_imag = tf.matmul(h4_real, SN(D_W5_imag,"sn5_real_imag")) + D_b5_imag
            tf_imag_imag = tf.matmul(h4_imag, SN(D_W5_imag,"sn5_imag_imag")) + D_b5_imag

            real_out = tf_real_real - tf_imag_imag
            imag_out = tf_imag_real + tf_real_imag
            
            prob = tf.complex(real_out, imag_out)

        else:
            D_W1 = tf.get_variable('D_W1', [x.get_shape()[1] + z.get_shape()[1], 512], initializer=w_init)
            D_b1 = tf.get_variable('D_b1', [512], initializer=b_init)
            inputs = tf.concat([x, z], axis=1)
            h1 = tf.nn.leaky_relu(tf.matmul(inputs, SN(D_W1,"sn1")) + D_b1)

            D_W2 = tf.get_variable('D_W2', [512, 256], initializer=w_init)
            D_b2 = tf.get_variable('D_b2', [256], initializer=b_init)
            h2 = tf.nn.leaky_relu(tf.matmul(h1, SN(D_W2,"sn2")) + D_b2)

            D_W3 = tf.get_variable('D_W3', [256, 128], initializer=w_init)
            D_b3 = tf.get_variable('D_b3', [128], initializer=b_init)
            h3 = tf.nn.leaky_relu(tf.matmul(h2, SN(D_W3,"sn3")) + D_b3)

            D_W4 = tf.get_variable('D_W4', [128, 64], initializer=w_init)
            D_b4 = tf.get_variable('D_b4', [64], initializer=b_init)
            h4 = tf.nn.leaky_relu(tf.matmul(h3, SN(D_W4,"sn4")) + D_b4)

            D_W5 = tf.get_variable('D_W5', [64, 1], initializer=w_init)
            D_b5 = tf.get_variable('D_b5', [1], initializer=b_init)
            prob = tf.matmul(h4, SN(D_W5,"sn5")) + D_b5

    return prob
