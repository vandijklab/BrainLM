import os
import argparse
from math import ceil

import numpy as np
import pandas as pd
from datasets import Dataset


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
    print("UKBioBank FMRI Data Arrow Conversion Starting...")
    # Assuming that filename is patient ID, thus each file with unique name is a separate patient.
    all_dat_files = os.listdir(args.uk_biobank_dir)
    all_dat_files = [filename for filename in all_dat_files if ".dat" in filename]
    all_dat_files.remove("A424_Coordinates.dat")
    all_dat_files.sort()  # Sorted in ascending order, first 80% will be train. Assuming no bias in patient order

    train_split_idx = ceil(len(all_dat_files) * 0.8)
    val_split_idx = ceil(len(all_dat_files) * 0.9)
    train_files = all_dat_files[:train_split_idx]
    val_files = all_dat_files[train_split_idx:val_split_idx]
    test_files = all_dat_files[val_split_idx:]

    # --- Normalization Calculations ---#
    # Calculate min and max value across train, validation, and test sets
    global_train_max = -1e9
    global_train_min = 1e9
    voxel_maximums_train = []
    voxel_minimums_train = []

    for filename in train_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        assert (
            np.min(dat_arr) >= 0
        ), "Minimum of patient recording is a negative number, check normalization"
        if np.max(dat_arr) > global_train_max:
            global_train_max = np.max(dat_arr)
        if np.min(dat_arr) < global_train_min:
            global_train_min = np.min(dat_arr)

        dat_arr_max = np.max(dat_arr, axis=0)
        dat_arr_min = np.min(dat_arr, axis=0)
        voxel_maximums_train.append(dat_arr_max)
        voxel_minimums_train.append(dat_arr_min)

    voxel_maximums_train = np.stack(voxel_maximums_train, axis=0)
    voxel_minimums_train = np.stack(voxel_minimums_train, axis=0)
    global_per_voxel_train_max = np.max(voxel_maximums_train, axis=0)
    global_per_voxel_train_min = np.min(voxel_minimums_train, axis=0)

    # Validation
    global_val_max = -1e9
    global_val_min = 1e9
    voxel_maximums_val = []
    voxel_minimums_val = []

    for filename in val_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        assert (
            np.min(dat_arr) >= 0
        ), "Minimum of patient recording is a negative number, check normalization"
        if np.max(dat_arr) > global_val_max:
            global_val_max = np.max(dat_arr)
        if np.min(dat_arr) < global_val_min:
            global_val_min = np.min(dat_arr)

        dat_arr_max = np.max(dat_arr, axis=0)
        dat_arr_min = np.min(dat_arr, axis=0)
        voxel_maximums_val.append(dat_arr_max)
        voxel_minimums_val.append(dat_arr_min)

    voxel_maximums_val = np.stack(voxel_maximums_val, axis=0)
    voxel_minimums_val = np.stack(voxel_minimums_val, axis=0)
    global_per_voxel_val_max = np.max(voxel_maximums_val, axis=0)
    global_per_voxel_val_min = np.min(voxel_minimums_val, axis=0)

    # Test
    global_test_max = -1e9
    global_test_min = 1e9
    voxel_maximums_test = []
    voxel_minimums_test = []

    for filename in test_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        assert (
            np.min(dat_arr) >= 0
        ), "Minimum of patient recording is a negative number, check normalization"
        if np.max(dat_arr) > global_test_max:
            global_test_max = np.max(dat_arr)
        if np.min(dat_arr) < global_test_min:
            global_test_min = np.min(dat_arr)

        dat_arr_max = np.max(dat_arr, axis=0)
        dat_arr_min = np.min(dat_arr, axis=0)
        voxel_maximums_test.append(dat_arr_max)
        voxel_minimums_test.append(dat_arr_min)

    voxel_maximums_test = np.stack(voxel_maximums_test, axis=0)
    voxel_minimums_test = np.stack(voxel_minimums_test, axis=0)
    global_per_voxel_test_max = np.max(voxel_maximums_test, axis=0)
    global_per_voxel_test_min = np.min(voxel_minimums_test, axis=0)

    # --- Convert All .dat Files to Arrow Datasets ---#
    # Training set
    train_dataset_dict = {
        "Raw_Recording": [],
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

    for filename in train_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        global_norm_dat_arr = np.copy(dat_arr)
        per_patient_all_voxels_norm_dat_arr = np.copy(dat_arr)
        per_patient_per_voxel_norm_dat_arr = np.copy(dat_arr)
        per_voxel_all_patient_norm_dat_arr = np.copy(dat_arr)
        recording_mean_subtracted = np.copy(dat_arr)
        recording_mean_subtracted2 = np.copy(dat_arr)
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

        _99th_global_recording = np.divide(recording_mean_subtracted2, _99th_percentile)

        train_dataset_dict["Raw_Recording"].append(dat_arr)
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
        dataset_path=os.path.join(save_path, "train_ukbiobank1000")
    )

    # Val set
    val_dataset_dict = {
        "Raw_Recording": [],
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
    for filename in val_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        global_norm_dat_arr = np.copy(dat_arr)
        per_patient_all_voxels_norm_dat_arr = np.copy(dat_arr)
        per_patient_per_voxel_norm_dat_arr = np.copy(dat_arr)
        per_voxel_all_patient_norm_dat_arr = np.copy(dat_arr)
        recording_mean_subtracted = np.copy(dat_arr)
        recording_mean_subtracted2 = np.copy(dat_arr)
        global_std = 41.44047  # calculated in normalization notebook
        _99th_percentile = 111.13143061224855  # calculated externally

        # All patients, all voxels normalization
        if (global_val_max - global_val_min) > 0.0:
            global_norm_dat_arr = (global_norm_dat_arr - global_val_min) / (
                global_val_max - global_val_min
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
            voxel_maximum = global_per_voxel_val_max[voxel_idx]
            voxel_minimum = global_per_voxel_val_min[voxel_idx]
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

        _99th_global_recording = np.divide(recording_mean_subtracted2, _99th_percentile)

        val_dataset_dict["Raw_Recording"].append(dat_arr)
        val_dataset_dict["All_Patient_All_Voxel_Normalized_Recording"].append(
            global_norm_dat_arr
        )
        val_dataset_dict["Per_Patient_All_Voxel_Normalized_Recording"].append(
            per_patient_all_voxels_norm_dat_arr
        )
        val_dataset_dict["Per_Patient_Per_Voxel_Normalized_Recording"].append(
            per_patient_per_voxel_norm_dat_arr
        )
        val_dataset_dict["Per_Voxel_All_Patient_Normalized_Recording"].append(
            per_voxel_all_patient_norm_dat_arr
        )
        val_dataset_dict["Subtract_Mean_Normalized_Recording"].append(
            recording_mean_subtracted
        )
        val_dataset_dict["Subtract_Mean_Divide_Global_STD_Normalized_Recording"].append(
            z_score_global_recording
        )
        val_dataset_dict[
            "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording"
        ].append(_99th_global_recording)
        val_dataset_dict["Filename"].append(filename)
        val_dataset_dict["Patient ID"].append(filename.split(".dat")[-1])

    arrow_val_dataset = Dataset.from_dict(val_dataset_dict)
    arrow_val_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "val_ukbiobank1000")
    )

    # Test set
    test_dataset_dict = {
        "Raw_Recording": [],
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
    for filename in test_files:
        dat_arr = np.loadtxt(os.path.join(args.uk_biobank_dir, filename)).astype(
            np.float32
        )
        global_norm_dat_arr = np.copy(dat_arr)
        per_patient_all_voxels_norm_dat_arr = np.copy(dat_arr)
        per_patient_per_voxel_norm_dat_arr = np.copy(dat_arr)
        per_voxel_all_patient_norm_dat_arr = np.copy(dat_arr)
        recording_mean_subtracted = np.copy(dat_arr)
        global_std = 41.44047  # calculated in normalization notebook
        recording_mean_subtracted2 = np.copy(dat_arr)
        _99th_percentile = 111.13143061224855  # calculated externally

        # All patients, all voxels normalization
        if (global_test_max - global_test_min) > 0.0:
            global_norm_dat_arr = (global_norm_dat_arr - global_test_min) / (
                global_test_max - global_test_min
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
            voxel_maximum = global_per_voxel_test_max[voxel_idx]
            voxel_minimum = global_per_voxel_test_min[voxel_idx]
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

        _99th_global_recording = np.divide(recording_mean_subtracted2, _99th_percentile)

        test_dataset_dict["Raw_Recording"].append(dat_arr)
        test_dataset_dict["All_Patient_All_Voxel_Normalized_Recording"].append(
            global_norm_dat_arr
        )
        test_dataset_dict["Per_Patient_All_Voxel_Normalized_Recording"].append(
            per_patient_all_voxels_norm_dat_arr
        )
        test_dataset_dict["Per_Patient_Per_Voxel_Normalized_Recording"].append(
            per_patient_per_voxel_norm_dat_arr
        )
        test_dataset_dict["Per_Voxel_All_Patient_Normalized_Recording"].append(
            per_voxel_all_patient_norm_dat_arr
        )
        test_dataset_dict["Subtract_Mean_Normalized_Recording"].append(
            recording_mean_subtracted
        )
        test_dataset_dict[
            "Subtract_Mean_Divide_Global_STD_Normalized_Recording"
        ].append(z_score_global_recording)
        test_dataset_dict[
            "Subtract_Mean_Divide_Global_99thPercent_Normalized_Recording"
        ].append(_99th_global_recording)
        test_dataset_dict["Filename"].append(filename)
        test_dataset_dict["Patient ID"].append(filename.split(".dat")[-1])

    arrow_test_dataset = Dataset.from_dict(test_dataset_dict)
    arrow_test_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "test_ukbiobank1000")
    )

    # --- Save Brain Region Coordinates Into Another Arrow Dataset ---#
    coords_dat = np.loadtxt(
        os.path.join(args.uk_biobank_dir, "A424_Coordinates.dat")
    ).astype(np.float32)
    coords_pd = pd.DataFrame(coords_dat, columns=["Index", "X", "Y", "Z"])
    coords_dataset = Dataset.from_pandas(coords_pd)
    coords_dataset.save_to_disk(
        dataset_path=os.path.join(save_path, "Brain_Region_Coordinates")
    )
    print("Done.")


def main():
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument(
        "--uk_biobank_dir",
        type=str,
        default="datasets/UKBioBank10",
        help="Path to directory containing dat files, A424 coordinates file, and A424 excel sheet.",
    )
    parser.add_argument(
        "--arrow_dataset_save_directory",
        type=str,
        default="./datasets",
        help="The directory where you want to save the output arrow datasets.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="UKBioBank10_Arrow",
        help="The name of the arrow dataset which will be saved. This directory will be created, and"
        "will contain subfolders of arrow datasets. Each dat file will be converted into one "
        "arrow dataset.",
    )
    args = parser.parse_args()

    save_path = os.path.join(args.arrow_dataset_save_directory, args.dataset_name)
    assert not os.path.exists(save_path), "Specified arrow dataset already exists!"

    os.mkdir(save_path)
    convert_to_arrow_datasets(args, save_path)


if __name__ == "__main__":
    main()
