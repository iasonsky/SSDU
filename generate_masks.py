from masks import ssdu_masks
import h5py as h5

mask = ssdu_masks.ssdu_masks()
kspace_dir = "/data/projects/recon/data/public/fastmri/knees/PD/multicoil_train/file1000108.h5"
kspace_train = h5.File(kspace_dir, "r")['kspace'][:]

trn_mask, loss_mask = mask.Gaussian_selection(input_data=kspace_train, input_mask='')