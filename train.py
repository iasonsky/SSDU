# import tensorflow.compat.v1 as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow as tf

import scipy.io as sio
import numpy as np
import time
from datetime import datetime
import os
import h5py as h5
import utils
import tf_utils
import parser_ops
import masks.ssdu_masks as ssdu_masks
import UnrollNet
from utils import center_crop
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from sklearn.metrics import mean_squared_error

from tensorboardX import SummaryWriter

parser = parser_ops.get_parser()
args = parser.parse_args()
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"s

save_dir ='saved_models'
directory = os.path.join(save_dir, 'SSDU_' + args.data_opt + '_' +str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) + '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type+'Selection' )

if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize writer
writer = SummaryWriter(log_dir=directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#...............................................................................
start_time = time.time()
print('.................SSDU Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
kspace_dir, coil_dir, mask_dir = utils.get_train_directory(args)
# Take the fname in order to log the slices to tensorboard
fname = kspace_dir.split('/')[-1].split('.')[0]

# Load kspace, sensitivity maps and mask
kspace_train = h5.File(kspace_dir, "r")['kspace'][:]
sens_maps = h5.File(coil_dir, "r")['sensitivity_map'][:]
original_mask = h5.File(mask_dir, "r")['random1d'][:]

print('\n Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

nSlices, *_ = kspace_train.shape

trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)
ref_kspace = np.transpose(ref_kspace, (0, 3, 1, 2))

# center crop the kspace and sensitivity maps
imspace_train = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kspace_train, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
imspace_train = center_crop(imspace_train, [args.nrow_GLOB, args.ncol_GLOB])
kspace_train = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(imspace_train, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
sens_maps = center_crop(sens_maps, [args.nrow_GLOB, args.ncol_GLOB])
original_mask = np.repeat(np.expand_dims(original_mask, 0), args.nrow_GLOB, axis=0) # 2d projection of the 1d mask

# calculate target
target = np.abs(np.sum(imspace_train * np.conj(sens_maps), axis=1))

print('\n size of kspace: ', kspace_train.shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape, ', trn_mask: ', trn_mask.shape, ', loss_mask: ', loss_mask.shape, ', nw_input: ', nw_input.shape, ', ref_kspace: ', ref_kspace.shape)

print('\n create training and loss masks and generate network inputs... ')
ssdu_masker = ssdu_masks.ssdu_masks()
for ii in range(nSlices):
    if np.mod(ii, 50) == 0:
        print('\n Iteration: ', ii)

    if args.mask_type == 'Gaussian':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask, num_iter=ii)
    elif args.mask_type == 'Uniform':
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.uniform_selection(kspace_train[ii], original_mask, num_iter=ii)
    else:
        raise ValueError('Invalid mask selection')

    sub_kspace = kspace_train[ii] * trn_mask[ii][np.newaxis]
    ref_kspace[ii] = kspace_train[ii] * loss_mask[ii][np.newaxis]

    nw_input[ii] = utils.sense1(sub_kspace, sens_maps[ii, ...])

# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal in the training mask we set corresponding columns as 1 to ensure data consistency
if args.data_opt == 'Coronal_PD':
    trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

# %% Prepare the data for the training
ref_kspace = utils.complex2real(ref_kspace)
nw_input = utils.complex2real(nw_input)

# %% set the batch size
total_batch = int(np.floor(np.float32(nw_input.shape[0]) / (args.batchSize)))
kspaceP = tf.placeholder(tf.float32, shape=(None, None, None, None, 2), name='refkspace')
sens_mapsP = tf.placeholder(tf.complex64, shape=(None, None, None, None), name='sens_maps')
trn_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='trn_mask')
loss_maskP = tf.placeholder(tf.complex64, shape=(None, None, None), name='loss_mask')
nw_inputP = tf.placeholder(tf.float32, shape=(None, args.nrow_GLOB, args.ncol_GLOB, 2), name='nw_input')

# %% creating the dataset
dataset = tf.data.Dataset.from_tensor_slices((kspaceP, nw_inputP, sens_mapsP, trn_maskP, loss_maskP))
dataset = dataset.shuffle(buffer_size=10 * args.batchSize)
dataset = dataset.batch(args.batchSize)
dataset = dataset.prefetch(args.batchSize)
iterator = dataset.make_initializable_iterator()
ref_kspace_tensor, nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor = iterator.get_next('getNext')

# %% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model

scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))

all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)

saver = tf.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')

totalLoss = []
avg_cost = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize, ', Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    print('Training...')
    for ep in range(1, args.epochs + 1):
        print('Epoch: ', ep)
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        for jj in range(total_batch):
            print(f"\rBatch: {jj + 1}/{total_batch}", end="")
            tmp, _, _ = sess.run([loss, update_ops, optimizer])
            avg_cost += tmp / total_batch
        toc = time.time() - tic
        totalLoss.append(avg_cost)
        print(" Epoch:", ep, "elapsed_time = ""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))

        # log loss to tensorboard
        writer.add_scalar('Loss/train', avg_cost, ep)

        # add to tensorboard (every 10 epochs run the model for inference)
        if ep % 10 == 0:
            nw_output_ep, _, _, _, _ = UnrollNet.UnrolledNet(nw_inputP, sens_mapsP, trn_maskP, loss_maskP).model
            nw_output_ep = sess.run(nw_output_ep, feed_dict=feedDict)
            nw_output_ep = np.abs(nw_output_ep[..., 0])

            ssims = []
            psnrs = []
            mses = []

            for slice in range(nw_output_ep.shape[0]):
                target[slice] = target[slice] / np.max(target[slice])
                nw_output_ep[slice] = nw_output_ep[slice] / np.max(nw_output_ep[slice])

                # in order to log to tensorboard we have to expand the dimensions
                writer.add_image(f"{fname}_{str(slice)}/Mask_original", np.abs(np.expand_dims(original_mask, 0)).astype(np.float32), 0)
                writer.add_image(f"{fname}_{str(slice)}/Mask_training", np.abs(np.expand_dims(trn_mask[slice], 0)).astype(np.float32), 0)
                writer.add_image(f"{fname}_{str(slice)}/Mask_loss", np.abs(np.expand_dims(loss_mask[slice], 0)).astype(np.float32), 0)
                writer.add_image(f'{fname}_{str(slice)}/Target', np.expand_dims(target[slice], 0).astype(np.float32), ep)
                writer.add_image(f'{fname}_{str(slice)}/Reconstruction', np.expand_dims(nw_output_ep[slice], 0).astype(np.float32), ep)
                writer.add_image(f'{fname}_{str(slice)}/Error', np.expand_dims(np.abs(target[slice] - nw_output_ep[slice]), 0).astype(np.float32), ep)

                # calculate SSIM & PSNR between target and reconstruction
                ssim = structural_similarity(nw_output_ep[slice], target[slice])
                ssims.append(ssim)
                psnr = peak_signal_noise_ratio(nw_output_ep[slice], target[slice])
                psnrs.append(psnr)
                mse = mean_squared_error(nw_output_ep[slice], target[slice])
                mses.append(mse)

                # log SSIM & PSNR to tensorboard
                writer.add_scalar(f'SSIM/{fname}_{str(slice)}', ssim, ep)
                writer.add_scalar(f'PSNR/{fname}_{str(slice)}', psnr, ep)
                writer.add_scalar(f'MSE/{fname}_{str(slice)}', mse, ep)
            
            # import pdb
            # pdb.set_trace()
            # writer.add_scalar(f'vol_SSIM/{fname}_{str(slice)}', np.stack(ssims, 0), ep)
            # writer.add_scalar(f'vol_PSNR/{fname}_{str(slice)}', np.stack(psnrs, 0), ep)
            # writer.add_scalar(f'vol_NMSE/{fname}_{str(slice)}', np.stack(nmses, 0), ep)

            saver.save(sess, sess_trn_filename, global_step=ep)

end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
