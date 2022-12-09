from typing import Tuple
import os
import numpy as np
import tensorflow as tf
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py as h5
import time
import utils
import parser_ops
import pdb

def complex_center_crop(data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Apply a center crop to the input image or batch of complex images.
    Parameters
    ----------
    data: The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is
        applied along dimensions -3 and -2 and the last dimensions should have a size of 2.
    shape: The output shape. The shape should be smaller than the corresponding dimensions of data.
    Returns
    -------
    The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):       
        raise ValueError("Invalid shapes.")

    w_from = np.divide((data.shape[-3] - shape[0]), 2)
    h_from = np.divide((data.shape[-2] - shape[1]), 2)
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    # cast everything to int
    w_from = int(w_from)
    w_to = int(w_to)
    h_from = int(h_from)
    h_to = int(h_to)

    return data[:, w_from:w_to , h_from:h_to, :]  # type: ignore

def mask_center_crop(data: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Apply a center crop to the input image or batch of complex images.
    Parameters
    ----------
    data: The complex input tensor to be center cropped. It should have at least 3 dimensions and the cropping is
        applied along dimensions -3 and -2 and the last dimensions should have a size of 2.
    shape: The output shape. The shape should be smaller than the corresponding dimensions of data.
    Returns
    -------
    The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):       
        raise ValueError("Invalid shapes.")

    w_from = np.divide((data.shape[-2] - shape[0]), 2)
    h_from = np.divide((data.shape[-1] - shape[1]), 2)
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    # cast everything to int
    w_from = int(w_from)
    w_to = int(w_to)
    h_from = int(h_from)
    h_to = int(h_to)

    return data[w_from:w_to , h_from:h_to]  # type: ignore

parser = parser_ops.get_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# .......................Load the Data...........................................
print('\n Loading ' + args.data_opt + ' test dataset...')
kspace_dir, coil_dir, mask_dir, saved_model_dir = utils.get_test_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset
kspace_test = h5.File(kspace_dir, "r")['kspace'][:]
kspace_test = np.transpose(kspace_test, (0, 2, 3, 1))
print('kspace_test: ', kspace_test.shape)

# Transform the kspace in imspace in order to crop it
imspace_test = np.fft.fftshift(kspace_test, axes=(-3, -2))
imspace_test = np.fft.ifftn(imspace_test, axes=(-3, -2), norm="ortho")
imspace_test = np.fft.ifftshift(imspace_test, axes=(-3, -2))

# Plot imspace
plt.imshow(np.abs(imspace_test[20, :, :, 0]), cmap='gray')
plt.title('Imspace')
plt.show()

# Apply the cropping to imspace
imspace_test_cropped = complex_center_crop(imspace_test, (320,368))
print('imspace_test_cropped_cropped.shape:', imspace_test_cropped.shape)

# Plot cropped imspace
plt.imshow(np.abs(imspace_test_cropped[20, :, :, 0]), cmap='gray')
plt.title('Imspace Cropped')
plt.show()

# Transform imspace to kspace
kspace_test_cropped = np.fft.fftshift(imspace_test_cropped, axes=(-3, -2))
kspace_test_cropped = np.fft.fftn(kspace_test_cropped, axes=(-3, -2), norm="ortho")
kspace_test_cropped = np.fft.ifftshift(kspace_test_cropped, axes=(-3, -2))
print('kspace_test_cropped: ',kspace_test_cropped.shape)

# assign the cropped k-space
kspace_test = kspace_test_cropped 

# Load sensitivity maps
sens_maps_testAll = h5.File(coil_dir, "r")['sensitivity_map'][:]
sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 2, 3, 1))

# Crop sensitivity maps 
sens_maps_testAll = complex_center_crop(sens_maps_testAll, (320,368))
original_mask = h5.File(mask_dir, "r")['gaussian2d'][:]
original_mask = mask_center_crop(original_mask, (320,368)) # crop the mask

print('\n Normalize kspace to 0-1 region')
for ii in range(np.shape(kspace_test)[0]):
    kspace_test[ii, :, :, :] = kspace_test[ii, :, :, :] / np.max(np.abs(kspace_test[ii, :, :, :][:]))

# %% Train and loss masks are kept same as original mask during inference
nSlices, *_ = kspace_test.shape
test_mask = np.complex64(np.tile(original_mask[np.newaxis, :, :], (nSlices, 1, 1)))

print('\n size of kspace: ', kspace_test.shape, ', maps: ', sens_maps_testAll.shape, ', mask: ', test_mask.shape)

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    test_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    test_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

test_refAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
test_inputAll = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

print('\n generating the refs and sense1 input images')
for ii in range(nSlices):
    sub_kspace = kspace_test[ii] * np.tile(test_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    test_refAll[ii] = utils.sense1(kspace_test[ii, ...], sens_maps_testAll[ii, ...])
    test_inputAll[ii] = utils.sense1(sub_kspace, sens_maps_testAll[ii, ...])

sens_maps_testAll = np.transpose(sens_maps_testAll, (0, 3, 1, 2))
all_ref_slices, all_input_slices, all_recon_slices = [], [], []

print('\n  loading the saved model ...')
tf.reset_default_graph()
loadChkPoint = tf.train.latest_checkpoint(saved_model_dir)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    new_saver = tf.train.import_meta_graph(saved_model_dir + '/model_test.meta')
    new_saver.restore(sess, loadChkPoint)

    # ..................................................................................................................
    graph = tf.get_default_graph()
    nw_output = graph.get_tensor_by_name('nw_output:0')
    nw_kspace_output = graph.get_tensor_by_name('nw_kspace_output:0')
    mu_param = graph.get_tensor_by_name('mu:0')
    x0_output = graph.get_tensor_by_name('x0:0')
    all_intermediate_outputs = graph.get_tensor_by_name('all_intermediate_outputs:0')

    # ...................................................................................................................
    trn_maskP = graph.get_tensor_by_name('trn_mask:0')
    loss_maskP = graph.get_tensor_by_name('loss_mask:0')
    nw_inputP = graph.get_tensor_by_name('nw_input:0')
    sens_mapsP = graph.get_tensor_by_name('sens_maps:0')
    weights = sess.run(tf.global_variables())

    for ii in range(nSlices):

        ref_image_test = np.copy(test_refAll[ii, :, :])[np.newaxis]
        nw_input_test = np.copy(test_inputAll[ii, :, :])[np.newaxis]
        sens_maps_test = np.copy(sens_maps_testAll[ii, :, :, :])[np.newaxis]
        testMask = np.copy(test_mask[ii, :, :])[np.newaxis]
        ref_image_test, nw_input_test = utils.complex2real(ref_image_test), utils.complex2real(nw_input_test)

        tic = time.time()
        dataDict = {nw_inputP: nw_input_test, trn_maskP: testMask, loss_maskP: testMask, sens_mapsP: sens_maps_test}
        nw_output_ssdu, *_ = sess.run([nw_output, nw_kspace_output, x0_output, all_intermediate_outputs, mu_param], feed_dict=dataDict)
        toc = time.time() - tic
        ref_image_test = utils.real2complex(ref_image_test.squeeze())
        nw_input_test = utils.real2complex(nw_input_test.squeeze())
        nw_output_ssdu = utils.real2complex(nw_output_ssdu.squeeze())

        if args.data_opt == 'Coronal_PD':
            """window levelling in presence of fully-sampled data"""
            factor = np.max(np.abs(ref_image_test[:]))
        else:
            factor = 1

        ref_image_test = np.abs(ref_image_test) / factor
        nw_input_test = np.abs(nw_input_test) / factor
        nw_output_ssdu = np.abs(nw_output_ssdu) / factor

        # ...............................................................................................................
        all_recon_slices.append(nw_output_ssdu)
        all_ref_slices.append(ref_image_test)
        all_input_slices.append(nw_input_test)

        print('\n Iteration: ', ii, 'elapsed time %f seconds' % toc)

plt.figure()
slice_num = 5
plt.subplot(1, 3, 1), plt.imshow(np.abs(all_ref_slices[slice_num]), cmap='gray'), plt.title('ref')
plt.subplot(1, 3, 2), plt.imshow(np.abs(all_input_slices[slice_num]), cmap='gray'), plt.title('input')
plt.subplot(1, 3, 3), plt.imshow(np.abs(all_recon_slices[slice_num]), cmap='gray'), plt.title('recon')
plt.show()
