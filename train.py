from typing import Tuple
import matplotlib.pyplot as plt
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
import pdb
import wandb

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
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
wandb.init(project="SSDU")

save_dir ='/home/iskylitsis/scratch/models/ssdu/saved_models'
directory = os.path.join(save_dir, 'SSDU_' + args.data_opt + '_' +str(args.epochs)+'Epochs_Rate'+ str(args.acc_rate) +\
                         '_' + str(args.nb_unroll_blocks) + 'Unrolls_' + args.mask_type+'Selection' )

if not os.path.exists(directory):
    os.makedirs(directory)

print('\n create a test model for the testing')
test_graph_generator = tf_utils.test_graph(directory)

#...........................................................................d....
start_time = time.time()
print('.................SSDU Training.....................')
tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# .......................Load the Data..........................................
print('\n Loading ', args.data_opt, ' data, acc rate : ', args.acc_rate, ', mask type :', args.mask_type)
kspace_dir, coil_dir, mask_dir = utils.get_train_directory(args)

# %% kspace and sensitivity maps are assumed to be in .h5 format and mask is assumed to be in .mat
# Users can change these formats based on their dataset

# Load kspace and transpose it in order to be [ nSlices, nrow, ncol, ncoil ]
kspace_train = h5.File(kspace_dir, "r")['kspace'][()]
kspace_train = np.transpose(kspace_train, (0, 2, 3, 1))
print('kspace_train: ',kspace_train.shape)

# Transform the kspace in imspace in order to crop it
imspace_train = np.fft.fftshift(kspace_train, axes=(-3, -2))
imspace_train = np.fft.ifftn(imspace_train, axes=(-3, -2), norm="ortho")
imspace_train = np.fft.ifftshift(imspace_train, axes=(-3, -2))

# Plot imspace
plt.imshow(np.abs(imspace_train[20, :, :, 0]), cmap='gray')
plt.title('Imspace')
plt.savefig('imspace.png')
plt.show()

# Apply the cropping to imspace
imspace_train_cropped = complex_center_crop(imspace_train, (320,368))
print('imspace_train_cropped.shape:', imspace_train_cropped.shape)

# Plot cropped imspace
plt.imshow(np.abs(imspace_train_cropped[20, :, :, 0]), cmap='gray')
plt.title('Imspace Cropped')
plt.savefig('imspace_cropped.png')
plt.show()

# Transform imspace to kspace
kspace_train_cropped = np.fft.fftshift(imspace_train_cropped, axes=(-3, -2))
kspace_train_cropped = np.fft.fftn(kspace_train_cropped, axes=(-3, -2), norm="ortho")
kspace_train_cropped = np.fft.ifftshift(kspace_train_cropped, axes=(-3, -2))
print('kspace_train_cropped: ',kspace_train_cropped.shape)

# assign the cropped k-space
kspace_train = kspace_train_cropped 

# Load sensitivity maps and transpose them in order to be [ nSlices, nrow, ncol, ncoil ]
sens_maps = h5.File(coil_dir, "r")['sensitivity_map'][:]
sens_maps = np.transpose(sens_maps, (0, 2, 3, 1))
print('sensmaps shape: ',sens_maps.shape)
original_mask = h5.File(mask_dir, "r")['gaussian2d'][:]
print('original_mask.shape: ',original_mask.shape)
# original_mask = sio.loadmat(mask_dir)['mask']


# Plot sensitivity maps
plt.imshow(np.abs(sens_maps[20, :, :, 0]), cmap='gray')
plt.title('Sensitivity Map')
plt.savefig('sensmaps.png')
plt.show()

# Crop sensitivity maps 
sens_maps = complex_center_crop(sens_maps, (320,368))
print('sensmaps cropped shape: ',sens_maps.shape)
plt.imshow(np.abs(sens_maps[20, :, :, 0]), cmap='gray')
plt.title('Sensitivity Maps Cropped')
plt.savefig('sensmap_cropped.png')
plt.show()

# Plot original mask
plt.imshow(original_mask, cmap='gray')
plt.title('Original Mask')
plt.savefig('original_mask.png')
plt.show()

# Crop mask
original_mask = mask_center_crop(original_mask, (320,368))
print('original_mask cropped shape: ',original_mask.shape)
plt.imshow(np.abs(original_mask), cmap='gray')
plt.title('Original Mask Cropped')
plt.savefig('original_mask_cropped.png')
plt.show()

print('\n Normalize the kspace to 0-1 region')
for ii in range(np.shape(kspace_train)[0]):
    kspace_train[ii, :, :, :] = kspace_train[ii, :, :, :] / np.max(np.abs(kspace_train[ii, :, :, :][:]))

print('\n size of kspace: ', kspace_train.shape, ', maps: ', sens_maps.shape, ', mask: ', original_mask.shape)
nSlices, *_ = kspace_train.shape
print('nSlices= ', nSlices)

# # get the middle slice for kspace and sens maps
# kspace_train = kspace_train[kspace_train.shape[0] // 2]
# sens_maps = sens_maps[sens_maps.shape[0] // 2]

trn_mask, loss_mask = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64), \
                      np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)

nw_input = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB), dtype=np.complex64)
ref_kspace = np.empty((nSlices, args.nrow_GLOB, args.ncol_GLOB, args.ncoil_GLOB), dtype=np.complex64)

ref_kspace = np.zeros_like(kspace_train)
sub_kspace = np.zeros_like(kspace_train)

# pdb.set_trace()
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

    # plt.imshow(np.abs(trn_mask[0]), cmap='gray')
    # plt.savefig('train_mask.png')
    # print('\n kspace_train shape: ', kspace_train.shape, ',train mask shape: ', trn_mask.shape, ',loss mask shape: ', loss_mask.shape)
    # pdb.set_trace()
    sub_kspace = kspace_train[ii] * np.tile(trn_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    ref_kspace[ii, ...] = kspace_train[ii] * np.tile(loss_mask[ii][..., np.newaxis], (1, 1, args.ncoil_GLOB))
    nw_input[ii, ...] = utils.sense1(sub_kspace, sens_maps[ii, ...])

    # print(sub_kspace.shape, ref_kspace.shape, nw_input.shape)
    # pdb.set_trace()

    # # pdb.set_trace()
    # pdb.set_trace()
    # sub_kspace[ii] = kspace_train[ii] * np.expand_dims(trn_mask, 1)[ii]
    # print('sub_kspace.shape:',sub_kspace.shape)
    # ref_kspace[ii] = kspace_train[ii] * np.expand_dims(loss_mask, 1)[ii]
    # print('ref_kspace.shape:',sub_kspace.shape)
    # nw_input[ii] = utils.sense1(sub_kspace[ii], sens_maps[ii], axes=(-2,-1))
    # pdb.set_trace()
    
# %%  zeropadded outer edges of k-space with no signal- check github readme file for explanation for further explanations
# for coronal PD dataset, first 17 and last 16 columns of k-space has no signal
# in the training mask we set corresponding columns as 1 to ensure data consistency

# TODO: since we cropped before do we need this?
if args.data_opt == 'Coronal_PD':
    trn_mask[:, :, 0:17] = np.ones((nSlices, args.nrow_GLOB, 17))
    trn_mask[:, :, 352:args.ncol_GLOB] = np.ones((nSlices, args.nrow_GLOB, 16))

# %% Prepare the data for the training
sens_maps = np.transpose(sens_maps, (0, 3, 1, 2))
ref_kspace = utils.complex2real(np.transpose(ref_kspace, (0, 3, 1, 2)))
nw_input = utils.complex2real(nw_input)
# nw_input = utils.complex2real(np.transpose(nw_input), (0, 3, 1, 2))

# # %% Prepare the data for the training
# ref_kspace = utils.complex2real(ref_kspace)
# nw_input = utils.complex2real(nw_input)
# sens_maps = utils.complex2real(sens_maps)
# pdb.set_trace()
print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)
# pdb.set_trace()

# This gives OOM error
ref_kspace = ref_kspace[:5, ...]
nw_input = nw_input[:5, ...]
sens_maps = sens_maps[:5, ...]
trn_mask = trn_mask[:5, ...]
loss_mask = loss_mask[:5, ...]

# This runs but the cost/error = nan
# ref_kspace = ref_kspace[:15, :2, ...]
# nw_input = nw_input[:15, ...]
# sens_maps = sens_maps[:15, :2, ...]
# trn_mask = trn_mask[:15, ...]
# loss_mask = loss_mask[:15, ...]

print('\n size of ref kspace: ', ref_kspace.shape, ', nw_input: ', nw_input.shape, ', maps: ', sens_maps.shape, ', mask: ', trn_mask.shape)
# kspace shape = [slices, coils, x, y, 2] / if batch is on [batch_size, slices, coils, x, y, 2]

# # %% set the batch size 
# # TODO: we index the slices here, is this correct?
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
print(ref_kspace_tensor.shape, nw_input_tensor.shape, sens_maps_tensor.shape, trn_mask_tensor.shape, loss_mask_tensor.shape)
# pdb.set_trace()
# %% make training model
nw_output_img, nw_output_kspace, *_ = UnrollNet.UnrolledNet(nw_input_tensor, sens_maps_tensor, trn_mask_tensor, loss_mask_tensor).model 
scalar = tf.constant(0.5, dtype=tf.float32)
loss = tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace) / tf.norm(ref_kspace_tensor)) + \
       tf.multiply(scalar, tf.norm(ref_kspace_tensor - nw_output_kspace, ord=1) / tf.norm(ref_kspace_tensor, ord=1))
all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss)
saver = tf.train.Saver(max_to_keep=100)
sess_trn_filename = os.path.join(directory, 'model')
totalLoss = []
avg_cost = 0
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print('SSDU Parameters: Epochs: ', args.epochs, ', Batch Size:', args.batchSize,
          ', Number of trainable parameters: ', sess.run(all_trainable_vars))
    feedDict = {kspaceP: ref_kspace, nw_inputP: nw_input, trn_maskP: trn_mask, loss_maskP: loss_mask, sens_mapsP: sens_maps}

    print('Training...')
    # for ep in range(1, args.epochs + 1):
    for ep in range(1, 10):
        sess.run(iterator.initializer, feed_dict=feedDict)
        avg_cost = 0
        tic = time.time()
        try:

            for jj in range(total_batch):
                tmp, _, _ = sess.run([loss, update_ops, optimizer])
                avg_cost += tmp / total_batch
            toc = time.time() - tic
            totalLoss.append(avg_cost)
            print("Epoch:", ep, "elapsed_time =""{:f}".format(toc), "cost =", "{:.3f}".format(avg_cost))
            wandb.log({'loss': avg_cost})
        except tf.errors.OutOfRangeError:
            pass

        if (np.mod(ep, 10) == 0):
            saver.save(sess, sess_trn_filename, global_step=ep)
            sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
        wandb.tensorflow.log(tf.summary.merge_all())
end_time = time.time()
sio.savemat(os.path.join(directory, 'TrainingLog.mat'), {'loss': totalLoss})
wandb.log({'total_loss': totalLoss})
print('Training completed in  ', ((end_time - start_time) / 60), ' minutes')
