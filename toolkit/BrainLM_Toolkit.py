import numpy as np
import nibabel as nib
import os
import argparse
from math import ceil
import pickle

import pandas as pd
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import glob


def convert_fMRIvols_to_A424(data_path, output_path):
    """
    This function takes in a folder of preprocessed fMRI volumes (.nii.gz), extracts A424 parcels, and saves these
    timeseries data to .dat files. 
    
    Inputs:
        data_path: Directory of fMRI volumes
        output_path: Where to store the output parcellated time series (.dat files) 
    """
    
    # Where is the data located?
    paths = os.listdir(data_path)
    print("fMRI data path specified:", data_path)
    print("Number of fMRI files:", len(paths))

    # Atlas file (Standard space atlas with cortical GM expanded by 2mm in WM)
    l = './toolkit/atlases/A424+2mm.nii.gz'
    print("Atlas file:", l)
    try:
        label_img = nib.load(l)
        label = label_img.get_fdata()
        label = label.flatten()
        print("Atlas successfully loaded")
    except Exception as e:
        print(f'Loading dlabel Atlas File Error for {l}: {str(e)}')

    # Create fMRI .dat files
    for f in paths: # f = './fMRI.nii.gz'
        file_path = os.path.join(data_path,f)
        # Load images and labels
        if ".nii.gz" in f:
            print(f'Loading 4D image from {file_path}')

            try:
                dts_img = nib.load(file_path)
                dts = dts_img.get_fdata()
                print("Loaded fMRI data")

            except Exception as e:
                print(f'Loading 4D File Error for {f}: {str(e)}')

            try:
                print(f"Extracting A424 Parcels for {f}")
                m = dts.reshape((-1, dts.shape[-1])).T
                sh = m.shape[0]

                # Get parcellated time series given a label input
                # print(f'\nGet parcellated time series using {l}\n\n')

                nParcels = 424
                pMeas = np.zeros((nParcels, 3))
                pmTS = np.zeros((sh, nParcels))

                for i in range(1, nParcels + 1):
                    ind = (label == i)
                    y = m[:, ind]
                    pmTS[:, i - 1] = np.nanmean(y, axis=1)

                # Replace NaNs with 0
                pmTS[np.isnan(pmTS)] = 0

                # Save Time Series
                save_name = f.split('.nii.gz')[0]
                fn = os.path.join(output_path, f'{save_name}.dat')
                print(f"Saving file {fn} with shape {pmTS.shape} (TRs, parcels)")
                np.savetxt(fn, pmTS, delimiter='\t')

            except:
                print(f"Error with parcel Extraction for {f}")
        
        else:
            print(f"File {f} not a nifti file. Skipping...")

def convert_to_arrow_datasets(args, save_path):
    """
    This function accepts a arguments object containing the filepath of a directory containing
     dat files for patient fmri recordings from the UK BioBank dataset.
    This function will first create subdirectories for train, val, test, using a 800/100/100
     split for 1000 patients.
    Then, for each dataset split, each patient dat file ([timepoints x brain regions]) will be
    converted into its own arrow dataset.

    The arrow dataset will be saved to args.arrow_dataset_save_directory, and will be needed
    for HuggingFace training scripts for CellLM.

    Arguments:
        args: arguments object containing parameters:
            --uk_biobank_dir
            --arrow_dataset_save_directory
            --dataset_name
        save_path: concatenation of dataset save directory and arrow dataset name
    """
    # --- Train/val/test Split ---#
    print("FMRI Data Arrow Conversion Starting...")
    # Assuming that filename is patient ID, thus each file with unique name is a separate patient.
    all_dat_files = os.listdir(args["uk_biobank_dir"])
    all_dat_files = [filename for filename in all_dat_files if ".dat" in filename]
    try: 
        all_dat_files.remove("A424_Coordinates.dat")
        print('A424_Coordinates was removed from the list')
    except ValueError:
        print("There's no A24 Coordinates dat file")
    all_dat_files.sort()  # Sorted in ascending order, first 80% will be train. Assuming no bias in patient order

    train_split_idx = len(all_dat_files)
    train_files = all_dat_files[:train_split_idx]
    sh_35 = 0
    sh_less_200 = 0
    for idx,file in enumerate(tqdm(all_dat_files)):
        try:
            sample = np.loadtxt(os.path.join(args["uk_biobank_dir"],file)).T #490, 424
            if sample.shape[0] < 200:
                print(sample.shape, idx, "ommitted due to insufficient data")
                sh_less_200 += 1
            else:
                sh_35 += 1
            # print(sample.shape)
        except UnicodeDecodeError:
            print(file)
        

    print(f"Not processing {sh_less_200} files due to insufficient fMRI data")
    compute_Stats=True
    if compute_Stats: 
        num_files = sh_35 #len(all_dat_files_rs) + len(all_dat_files_tf)
        all_stds = np.zeros([num_files, 424]) 
        all_data = np.empty([num_files*200, 424])
        for idx,file in enumerate(tqdm(train_files)):
            if idx == num_files:
                break
            # if idx%2000==0:
            #     print('idx: {}, next file: {}'.format(idx,file))
            try:
                sample = np.loadtxt(os.path.join(args["uk_biobank_dir"],file)) #490, 424
                # print(sample.shape)
            except UnicodeDecodeError:
                print(file)
            # sample = np.loadtxt(os.path.join(uk_biobank_dir_rs, file, 'rfMRI_REST','rfMRI_REST_Atlas_MSMAll_hp2000_clean_MGTR_zscored_HCP_MMP_BNAC.dat')).astype(np.float32).T
            sample_mean = sample.mean(axis=0, keepdims=True)
            sample_mean = sample_mean[None,:].repeat(sample.shape[0],1).squeeze()
            sample = sample - sample_mean

            idx_sample=idx


            if sample.shape[0] < 200:
                continue
            try:
                all_data[idx*200:(idx+1)*200,:] = sample[:200,:]
            except ValueError:
                print(sample.shape)
                print('idx: {}, idx_sample: {}'.format(idx,idx_sample))

        global_std = np.std(all_data, axis=0) 
        data_median_per_voxel = np.median(all_data,axis=0)
        data_mean_per_voxel = np.mean(all_data,axis=0)

        all_data_nonzeros = np.copy(all_data)
        all_data_nonzeros[all_data_nonzeros == 0] = 'nan'
        quartiles = np.nanpercentile(all_data_nonzeros, [25, 75], axis=0)
        IQR = quartiles[1,:]-quartiles[0,:]


    # --- Normalization Calculations ---#
    # Calculate min and max value across train, validation, and test sets
    global_train_max = -1e9
    global_train_min = 1e9
    voxel_maximums_train = []
    voxel_minimums_train = []

    for filename in tqdm(train_files, desc="Getting normalization stats"):
        dat_arr = np.loadtxt(os.path.join(args["uk_biobank_dir"], filename)).astype(
            np.float32
        )
        #assert (
        #    np.min(dat_arr) >= 0
        #), "Minimum of patient recording is a negative number, check normalization"
        if np.max(dat_arr) > global_train_max:
            global_train_max = np.max(dat_arr)
        if np.min(dat_arr) < global_train_min:
            global_train_min = np.min(dat_arr)

        dat_arr_max = np.max(dat_arr, axis=1)
        dat_arr_min = np.min(dat_arr, axis=1)
        voxel_maximums_train.append(dat_arr_max)
        voxel_minimums_train.append(dat_arr_min)

    voxel_maximums_train = np.stack(voxel_maximums_train, axis=0)
    voxel_minimums_train = np.stack(voxel_minimums_train, axis=0)
    global_per_voxel_train_max = np.max(voxel_maximums_train, axis=0)
    global_per_voxel_train_min = np.min(voxel_minimums_train, axis=0)

    # --- Convert All .dat Files to Arrow Datasets ---#
    # Training set
    train_dataset_dict = {
        "Raw_Recording": [],
        "Voxelwise_RobustScaler_Normalized_Recording": [],
        "All_Patient_All_Voxel_Normalized_Recording": [],
        "Per_Patient_All_Voxel_Normalized_Recording": [],
        "Per_Patient_Per_Voxel_Normalized_Recording": [],
        "Per_Voxel_All_Patient_Normalized_Recording": [],
        "Subtract_Mean_Normalized_Recording": [],
        "Subtract_Mean_Divide_Global_STD_Normalized_Recording": [],
        "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording": [],
        "Filename": [],
        "Patient ID": [],
    }

    for filename in tqdm(train_files, desc="Normalizing Data"):
        dat_arr = np.loadtxt(os.path.join(args["uk_biobank_dir"], filename)).astype(
            np.float32
        ).T

        if dat_arr.shape[0] < 200:
            continue

        if dat_arr.shape[0] > 424:
            dat_arr = dat_arr[:350, :]

        global_norm_dat_arr = np.copy(dat_arr)
        per_patient_all_voxels_norm_dat_arr = np.copy(dat_arr)
        per_patient_per_voxel_norm_dat_arr = np.copy(dat_arr)
        per_voxel_all_patient_norm_dat_arr = np.copy(dat_arr)
        recording_mean_subtracted = np.copy(dat_arr)
        recording_mean_subtracted2 = np.copy(dat_arr)
        recording_mean_subtracted3 = np.copy(dat_arr.T)
        global_std = 41.44047  # calculated in normalization notebook
        _99th_percentile = 111.13143061224855  # calculated externally

        # All patients, all voxels normalization
        if (global_train_max - global_train_min) > 0.0:
            global_norm_dat_arr = (global_norm_dat_arr - global_train_min) / (
                global_train_max - global_train_min
            )

        # Per patient all voxel normalization
        patient_all_voxel_min_val = np.min(per_patient_all_voxels_norm_dat_arr)
        patient_all_voxel_max_val = np.max(per_patient_all_voxels_norm_dat_arr)
        if (patient_all_voxel_max_val - patient_all_voxel_min_val) > 0.0:
            per_patient_all_voxels_norm_dat_arr = (
                per_patient_all_voxels_norm_dat_arr - patient_all_voxel_min_val
            ) / (patient_all_voxel_max_val - patient_all_voxel_min_val)

        # Per patient per voxel normalization
        for voxel_idx in range(dat_arr.shape[1]):
            patient_voxel_min_val = per_patient_per_voxel_norm_dat_arr[
                :, voxel_idx
            ].min()
            patient_voxel_max_val = per_patient_per_voxel_norm_dat_arr[
                :, voxel_idx
            ].max()
            if (patient_voxel_max_val - patient_voxel_min_val) > 0.0:
                per_patient_per_voxel_norm_dat_arr[:, voxel_idx] = (
                    per_patient_per_voxel_norm_dat_arr[:, voxel_idx]
                    - patient_voxel_min_val
                ) / (patient_voxel_max_val - patient_voxel_min_val)

        # Per voxel all patient normalization
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_maximum = global_per_voxel_train_max[voxel_idx]
            voxel_minimum = global_per_voxel_train_min[voxel_idx]
            if (voxel_maximum - voxel_minimum) > 0.0:
                per_voxel_all_patient_norm_dat_arr[:, voxel_idx] = (
                    per_voxel_all_patient_norm_dat_arr[:, voxel_idx] - voxel_minimum
                ) / (voxel_maximum - voxel_minimum)

        # Subtract Mean, Scale by Global Standard Deviation normalization
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_mean = recording_mean_subtracted[:, voxel_idx].mean()
            recording_mean_subtracted[:, voxel_idx] = (
                recording_mean_subtracted[:, voxel_idx] - voxel_mean
            )

        z_score_global_recording = np.divide(recording_mean_subtracted, global_std)

        # Subtract Mean, Scale by global 99th percentile
        for voxel_idx in range(dat_arr.shape[1]):
            voxel_mean = recording_mean_subtracted2[:, voxel_idx].mean()
            recording_mean_subtracted2[:, voxel_idx] = (
                recording_mean_subtracted2[:, voxel_idx] - voxel_mean
            )
        
        #Voxelwise Robust Scaler Normalization
        recording_mean_subtracted3 = recording_mean_subtracted3 - recording_mean_subtracted3.mean(axis=0)
        recording_mean_subtracted3 = (recording_mean_subtracted3 - data_median_per_voxel / IQR)

        _99th_global_recording = np.divide(recording_mean_subtracted2, _99th_percentile)

        train_dataset_dict["Raw_Recording"].append(dat_arr)
        train_dataset_dict["Voxelwise_RobustScaler_Normalized_Recording"].append(recording_mean_subtracted3)
        train_dataset_dict["All_Patient_All_Voxel_Normalized_Recording"].append(
            global_norm_dat_arr
        )
        train_dataset_dict["Per_Patient_All_Voxel_Normalized_Recording"].append(
            per_patient_all_voxels_norm_dat_arr
        )
        train_dataset_dict["Per_Patient_Per_Voxel_Normalized_Recording"].append(
            per_patient_per_voxel_norm_dat_arr
        )
        train_dataset_dict["Per_Voxel_All_Patient_Normalized_Recording"].append(
            per_voxel_all_patient_norm_dat_arr
        )
        train_dataset_dict["Subtract_Mean_Normalized_Recording"].append(
            recording_mean_subtracted
        )
        train_dataset_dict[
            "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
        ].append(z_score_global_recording)
        train_dataset_dict[
            "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording"
        ].append(_99th_global_recording)
        train_dataset_dict["Filename"].append(filename)
        train_dataset_dict["Patient ID"].append(filename.split(".dat")[-1])

    arrow_train_dataset = Dataset.from_dict(train_dataset_dict)
    arrow_train_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "train")
    )

    # --- Save Brain Region Coordinates Into Another Arrow Dataset ---#
    coords_dat = np.loadtxt(os.path.join("/home/mt2286/project/BrainLM/toolkit/atlases/", "A424_Coordinates.dat")).astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "Brain_Region_Coordinates")
    )
    print("Done.")