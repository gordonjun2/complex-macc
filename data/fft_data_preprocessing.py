import scipy.io
import numpy as np
import os

###################################################################################
### Input Parameters Range (to scale data based on the min and max values used) ###
###################################################################################

## % sensor variables

# zc_min = 200000            # in m
# zc_max = 600000            # in m
# gz_ang_deg_min = 30
# gz_ang_deg_max = 80
# sq_ang_deg_min = 0
# sq_ang_deg_max = 90
# fc_min = 4e9
# fc_max = 15e9
# sr_min = 300
# sr_max = 700
# sa_min = 300
# sa_max = 700
# x_px_min = 0.5
# x_px_max = 2
# y_px_min = 0.5
# y_px_max = 2
# x_off_min = -50
# x_off_max = 50
# y_off_min = -50
# y_off_max = 50

RJ_max = 1.200000000397223e+006
RJ_min = 2.030853223768550e+005
phiJ_max = 80
phiJ_min = 30
squintJ_max = 90
squintJ_min = 0
vJ_max = 7.784260541829205e+003
vJ_min = 7.557864133681774e+003
fc_max = 15e9
fc_min = 4e9
prf_max = 9.205568633423554e+002
prf_min = 40.33651378211401
x_px_min = 0.5
x_px_max = 2
y_px_min = 0.5
y_px_max = 2
x_off_min = -50
x_off_max = 50
y_off_min = -50
y_off_max = 50

###################################################################################
###################################################################################
###################################################################################

ft_fft_mat_dir = '../ft_fft10/'
ft_fft_mat_files_list = os.listdir(ft_fft_mat_dir)

input_params_mat_dir = '../input_parameters/'
input_params_mat_files_list = os.listdir(input_params_mat_dir)

ft_fft_npy_dir = './fft-scattering-coef-40k/fft40K_images/'
input_params_npy_dir = './fft-scattering-coef-40k/fft40K_params/'

if not os.path.exists(ft_fft_npy_dir):
    os.makedirs(ft_fft_npy_dir)

if not os.path.exists(input_params_npy_dir):
    os.makedirs(input_params_npy_dir)

ft_fft_npy = np.array([]).reshape(0, 64*64*16)                      # 64*64*16 = 65536
input_params_npy = np.array([]).reshape(0, 10)

count = 0
save_num = 0

for ft_fft_mat in ft_fft_mat_files_list:

    count = count + 1
    if count % 100 == 0:
        print('Processing images data: ' + str(count) + ' / ' + str(len(ft_fft_mat_files_list)))

    mat = scipy.io.loadmat(ft_fft_mat_dir + ft_fft_mat)
    ft_fft = mat['ft_fft10_save_var']
    fft_reshaped = np.reshape(np.transpose(ft_fft, (1, 2, 0)), (1, -1), 'C')
    ft_fft_npy = np.vstack([ft_fft_npy, fft_reshaped])

    if count % 100 == 0:
        save_num = save_num + 1
        np.save(ft_fft_npy_dir + 'fft40K_images_' + str(save_num) + '.npy', ft_fft_npy)
        print('Saved ' + str(save_num) + ' out of ' + str(int(len(ft_fft_mat_files_list) / 100)) + ' parts of fft40K_images\n')
        ft_fft_npy = np.array([]).reshape(0, 64*64*16)

count = 0
save_num = 0

for input_params_mat in input_params_mat_files_list:

    count = count + 1
    if count % 100 == 0:
        print('Processing images data: ' + str(count) + ' / ' + str(len(input_params_mat_files_list)))

    mat = scipy.io.loadmat(input_params_mat_dir + input_params_mat)

    RJ = mat['RJ_save_var'][0]
    RJ_scaled = (RJ - RJ_min) / (RJ_max - RJ_min)

    phiJ = mat['phiJ_save_var'][0]
    phiJ_scaled = (phiJ - phiJ_min) / (phiJ_max - phiJ_min)

    squintJ = mat['squintJ_save_var'][0]
    squintJ_scaled = (squintJ - squintJ_min) / (squintJ_max - squintJ_min)

    vJ = mat['vJ_save_var'][0]
    vJ_scaled = (vJ - vJ_min) / (vJ_max - vJ_min)

    fc = mat['fc_save_var'][0]
    fc_scaled = (fc - fc_min) / (fc_max - fc_min)

    prf = mat['prf_save_var'][0]
    prf_scaled = (prf - prf_min) / (prf_max - prf_min)

    x_px = mat['x_px_save_var'][0]
    x_px_scaled = (x_px - x_px_min) / (x_px_max - x_px_min)

    y_px = mat['y_px_save_var'][0]
    y_px_scaled = (y_px - y_px_min) / (y_px_max - y_px_min)

    x_off = mat['x_off_save_var'][0]
    x_off_scaled = (x_off - x_off_min) / (x_off_max - x_off_min)

    y_off = mat['y_off_save_var'][0]
    y_off_scaled = (y_off - y_off_min) / (y_off_max - y_off_min)

    input_params = np.concatenate((RJ_scaled, phiJ_scaled, squintJ_scaled, vJ_scaled, fc_scaled, prf_scaled, x_px_scaled, y_px_scaled, x_off_scaled, y_off_scaled))
    input_params_npy = np.vstack([input_params_npy, input_params])

    if count % 100 == 0:
        save_num = save_num + 1
        np.save(input_params_npy_dir + 'fft40K_params_' + str(save_num) + '.npy', input_params_npy)
        print('Saved ' + str(save_num) + ' out of ' + str(int(len(input_params_mat_files_list) / 100)) + ' parts of fft40K_params\n')
        input_params_npy = np.array([]).reshape(0, 10)




