from masks import ssdu_masks
import h5py as h5
import numpy as np
from utils import center_crop
import pdb
import os 
import time

dataset_dir = "/data/projects/recon/data/public/fastmri/knees/PD/multicoil_train/"
output_filename = "/home/iskylitsis/scratch/data/ssdu_masks_output_all.h5"
# create h5 file 
output = h5.File(output_filename, 'w')
start_time = time.time()
for filename in os.listdir(dataset_dir):
    kspace_dir = dataset_dir + filename
    mask_dir = "masks/masks.h5"
    nrow_GLOB = 320
    ncol_GLOB = 368
    kspace_train = h5.File(kspace_dir, "r")['kspace'][:]
    original_mask = h5.File(mask_dir, "r")['random1d'][:]

    if not (0 < nrow_GLOB <= kspace_train.shape[-2] and 0 < ncol_GLOB <= kspace_train.shape[-1]):
        print("\nInvalid shapes.")
        continue

    # center crop the kspace and sensitivity maps
    imspace_train = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(kspace_train, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    imspace_train = center_crop(imspace_train, [nrow_GLOB, ncol_GLOB])
    kspace_train = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(imspace_train, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
    original_mask = np.repeat(np.expand_dims(original_mask, 0), nrow_GLOB, axis=0) # 2d projection of the 1d mask

    nSlices, *_ = kspace_train.shape

    trn_mask, loss_mask = np.empty((nSlices, nrow_GLOB, ncol_GLOB), dtype=np.complex64), np.empty((nSlices, nrow_GLOB, ncol_GLOB), dtype=np.complex64)
    
    print('\n create training and loss masks')
    ssdu_masker = ssdu_masks.ssdu_masks()
    group = output.create_group(kspace_dir.split('/')[-1][:-3] )
    for ii in range(nSlices):
        print('\n Slice: ', ii)
        
        trn_mask[ii, ...], loss_mask[ii, ...] = ssdu_masker.Gaussian_selection(kspace_train[ii], original_mask, num_iter=ii)
        subgrp = group.create_group("slice" + str(ii))
        subgrp.create_dataset('trn_mask', data=trn_mask)
        subgrp.create_dataset('loss_mask', data=loss_mask)
# close hdf5 file
output.close()
end_time = time.time()
print('Masks generated in  ', ((end_time - start_time) / 60), ' minutes')
