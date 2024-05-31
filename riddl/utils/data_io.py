# © 2024. Triad National Security, LLC. All rights reserved.
"""
Build training and tesing tensors for TensorFlow 2

"""
import os
import numpy as np
import pandas
import regex
import tensorflow as tf

from functools import reduce

import matplotlib.pyplot as plt

from obspy import read as obspy_read

station_info = [['I57', 7],
                ['I53', 8],
                ['I56', 4]]

###################################
# Methods to build and use
# tensor data for TensorFlow
###################################
# Normalizing function
def map_range(column):
    y_max = np.max(np.abs(column))
    y = column/y_max
    return y


def load_raw_fk_results(beam_file):
    "Extract F-statistic, back azimuth, and trace-velocity values from an InfraPy fk results file."
    dat = pandas.read_csv(beam_file, skiprows=34, header=None, delimiter=' ')
    dat.columns = ['Time [sec]', 'Baz', 'Tr_vel', 'F']
    trace_vel = dat['Tr_vel']
    baz = dat['Baz']
    F_stat = dat['F']

    return F_stat, baz, trace_vel


def get_file_list(data_dir: str, file_pattern: list = None, file_type: str = '.npy') -> list:
    " Return a list of files in the data directory. Option to filter by a file pattern list."
    if file_pattern is not None:
        # Filter by station names
        if type(file_pattern) == str:
            files = [data_dir + f for f in os.listdir(data_dir) if (regex.search(file_type, f) is not None) and (regex.search(file_pattern, f) is not None)]
        else:
            files = []
            for item in file_pattern:
                files += [data_dir + f for f in os.listdir(data_dir) if (regex.search(file_type, f) is not None) and (regex.search(item, f) is not None)]
    else:
        # No station name filter
        files = [data_dir + f for f in os.listdir(data_dir) if regex.search(file_type, f) is not None]
    return files


def build_file_list(data_dir: str, labels: list, folders: list = [], file_pattern: list = None, file_type: str = '.npy') -> tuple:
    """
        Build a list of files..

    Args:
        data_dir (str):
        labels:
        folders:
        stations:
        pattern (str):

    Returns:
        sorted_files:
        counts:
    """

    # Formatting check
    if data_dir[-1] != "/":
        data_dir += "/"

    # Check if any files in the existing directory
    files = get_file_list(data_dir, file_pattern=file_pattern, file_type=file_type,)
    # Crawl through sub directories
    for folder in folders:
        data_temp = data_dir + folder + "/"
        files += get_file_list(data_temp, file_pattern=file_pattern, file_type=file_type)

    # Count files as a check; Parse the file names and count the number of each label
    separated_files = [[file for file in files if label in file] for label in labels]
    counts = [len(label_files) for label_files in separated_files]

    # Merge separated files so labelled files are grouped together
    sorted_files = [file for label_files in separated_files for file in label_files]

    print("File list summary:")
    for jj in range(0, len(labels)):
        print('\t' + str(int(counts[jj])) + " " + labels[jj] + " samples.")
    print(" ")

    return sorted_files, counts


def prep_fk_results(fk_peaks):
    """ Standardize (zero mean, unit standard dev.) fk results.
    """
    # Convert trace velocity and back azimuth to S_x and S_y
    sy = np.sin(np.radians(fk_peaks[1, :])) / fk_peaks[2, :]
    sx = np.cos(np.radians(fk_peaks[1, :])) / fk_peaks[2, :]

    sy = map_range(sy)
    sx = map_range(sx)
    F_stat = map_range(fk_peaks[3, :])

    # Re-stack F-statistic values, S_x, and S_y
    # result = np.column_stack((fk_peaks[3, :], sx, sy))

    # mean = np.mean(result, axis=0, keepdims=True)
    # stdev = np.std(result, axis=0, keepdims=True)

    # Standardize data to mean = 0 and standard deviation = 1.0
    # return (result - mean) / stdev

    return np.column_stack((F_stat, sx, sy))


def write_tensor_data(input_files: list, output_id: str, labels: list, test_fraction: float = 0.0, merge_labels: bool = None, balance: bool = True) -> None:
    """
    Build, balance, and split data from a list of fk_data files and save the results.

    Args:
        input_files (list): A list of fk files
        output_id (str): A pattern for the name of the output tensors
        labels (list): The...
        test_fraction (float): The fraction (0.0, 1.0) of data to assign for testing
        merge_labels:
        balance (bool): Balance the classes?

    Returns:
        Writes data to tensors.
    """
    # Check file count and data shape to define X and Y shapes
    label_cnt = len(labels)
    file_cnt = len(input_files)
    # m rows = (time, fk, baz, tr_vel)
    # n data points
    m, n = np.load(input_files[0], allow_pickle=True).shape
    # Ignore time row
    X = np.zeros((file_cnt, n, (m - 1)))
    Y = np.zeros(file_cnt, dtype=int)

    # Load data and convert labels to integers
    for jj, file in enumerate(input_files):
        arr = np.load(file, allow_pickle=True)
        if np.shape(arr)[1] != n:
            print("Warning! File length doesn't match: " + file)
        
        X[jj, :, :] = prep_fk_results(arr)
        Y[jj] = [idx for idx, label in enumerate(labels) if label in file][0]

    # Normalize
    for jj in range(0, file_cnt):
        for kk in range(0, (m - 1)):
            X[jj, :, kk] /= np.max(np.abs(X[jj, :, kk]))

    # Allocate new lists
    X_temp = [[Xn for idx, Xn in enumerate(X) if Y[idx] == jj] for jj in range(label_cnt)]
    Y_temp = [[Yn for Yn in Y if Yn == jj] for jj in range(label_cnt)]
    input_files_temp = [[fn for idx, fn in enumerate(input_files) if Y[idx] == jj] for jj in range(label_cnt)]

    # Merge labels and lower indices if there's a gap (e.g., avoid having labels [1, 3] and shift down to [0, 1])
    if merge_labels is not None:
        labels_remove = []
        print("Merging labels ", merge_labels, "...")
        # Determine if we're merging multiple categories
        if any(isinstance(item, list) for item in merge_labels):
            for jj in range(0, len(merge_labels)):
                merge_labels_temp = merge_labels[jj]
                merge_indices = [idx for idx, label in enumerate(labels) if label in merge_labels_temp]
                Y_temp[merge_indices[0]] = Y_temp[merge_indices[0]] + [Y_temp[merge_indices[0]][0]]*len(Y_temp[merge_indices[1]])
                Y_temp[merge_indices[1]].clear()
                X_temp[merge_indices[0]] = X_temp[merge_indices[0]] + X_temp[merge_indices[1]]
                X_temp[merge_indices[1]].clear()
                input_files_temp[merge_indices[0]] = input_files_temp[merge_indices[0]] + input_files_temp[merge_indices[1]]
                input_files_temp[merge_indices[1]].clear()
                labels_remove.append(labels[merge_indices[1]])
        else:
            merge_indices = [idx for idx, label in enumerate(labels) if label in merge_labels]
            Y_temp[merge_indices[0]] = Y_temp[merge_indices[0]] + [Y_temp[merge_indices[0]][0]]*len(Y_temp[merge_indices[1]])
            Y_temp[merge_indices[1]].clear()
            X_temp[merge_indices[0]] = X_temp[merge_indices[0]] + X_temp[merge_indices[1]]
            X_temp[merge_indices[1]].clear()
            input_files_temp[merge_indices[0]] = input_files_temp[merge_indices[0]] + input_files_temp[merge_indices[1]]
            input_files_temp[merge_indices[1]].clear()
            labels_remove.append(labels[merge_indices[1]])

        # Trim empty lists
        Y_temp = [item for item in Y_temp if item]
        X_temp = [item for item in X_temp if item]
        input_files_temp = [item for item in input_files_temp if item]
        labels = [item for item in labels if item not in labels_remove]
        print("Labels after merging are: " + str(labels))

        # Remove any labelling gaps
        y_list = np.arange(0, len(Y_temp))
        for jj in range(len(Y_temp)):
            if Y_temp[jj][0] >= y_list[jj]:
                Y_temp[jj] = [y_list[jj]] * len(Y_temp[jj])

    # Balance the data set so each label has the same count
    if balance is not None:
        print("Balancing data...")

        bal_X = []
        bal_Y = []
        bal_files = []
    
        min_count = np.min(np.array([len(item) for item in Y_temp]))
        print("min_count = " + str(min_count))

        for jj in range(0, len(Y_temp)):
            print('\t' + "Including " + str(min_count) + " " + labels[jj] + " samples...")
            permut = np.random.permutation(len(Y_temp[jj])).astype(int)[0:min_count]
            bal_X += [X_temp[jj][item] for item in permut]
            bal_Y += [Y_temp[jj][item] for item in permut]
            bal_files += [input_files_temp[jj][item] for item in permut]

        Y_temp = bal_Y
        X_temp = bal_X
        input_files_temp = bal_files

    # Convert lists to arrays
    Y = np.array(Y_temp)
    X = np.array(X_temp)
    input_files = np.array(input_files_temp)
    file_cnt = len(input_files) # Update

    # One-hot label info
    # https://www.tensorflow.org/api_docs/python/tf/one_hot
    Y = tf.one_hot(Y, depth=len(labels))

    print(" ")

    if (test_fraction != 0.0):
        print("Shuffling and splitting data into training / testing subsets using a test fraction = " + str(test_fraction))

        test_cnt = int(np.floor(file_cnt * test_fraction))
        train_cnt = file_cnt - test_cnt

        print('\t' + "Training sample count: " + str(train_cnt) + "/" + str(file_cnt))
        print('\t' + "Test sample count: " + str(test_cnt) + "/" + str(file_cnt))

        permut = np.random.permutation(range(file_cnt)).astype(int)
        idx_train = permut[0:train_cnt]
        idx_test = permut[train_cnt:]

        X_train = [X[idx] for idx in idx_train]
        Y_train = [Y[idx] for idx in idx_train]
        input_files_train = [input_files[idx] for idx in idx_train]

        X_test = [X[idx] for idx in idx_test]
        Y_test = [Y[idx] for idx in idx_test]
        input_files_test = [input_files[idx] for idx in idx_test]

        # Training Data
        data_array = np.zeros((len(input_files_train), 3), dtype='object')
        for jj, file in enumerate(input_files_train):
            arr = np.load(file, allow_pickle=True)
            file = file.split('/')[-1]
            name = file.split('.')[1]
            data_array[jj, 0] = name
            data_array[jj, 1] = str(np.max(arr[3, :]))
            data_array[jj, 2] = file.split('_')[-1][:-4]

        np.savez(output_id + "_train.X.npz", X_train)
        np.savez(output_id + "_train.Y.npz", Y_train)
        np.savetxt(output_id + "_train.file_list.txt", input_files_train, fmt="%s")
        np.savetxt(output_id + "_train.station_info.txt", data_array, fmt="%s")

        # Testing Data
        data_array = np.zeros((len(input_files_test), 3), dtype='object')
        for jj, file in enumerate(input_files_test):
            arr = np.load(file, allow_pickle=True)
            file = file.split('/')[-1]
            name = file.split('.')[1]
            data_array[jj, 0] = name
            data_array[jj, 1] = str(np.max(arr[3, :]))
            data_array[jj, 2] = file.split('_')[-1][:-4]

        np.savez(output_id + "_test.X.npz", X_test)
        np.savez(output_id + "_test.Y.npz", Y_test)
        np.savetxt(output_id + "_test.file_list.txt", input_files_test, fmt="%s")
        np.savetxt(output_id + "_test.station_info.txt", data_array, fmt="%s")

    else:
        print("Writing data without splitting...")
        # Save the infrasound array info
        data_array = np.zeros((len(input_files), 3), dtype='object')

        # Loop through and write station, channel count, max F-stat, and label
        for jj, file in enumerate(input_files):
            arr = np.load(file, allow_pickle=True)
            file = file.split('/')[-1]
            name = file.split('.')[1]
            data_array[jj, 0] = name
            data_array[jj, 1] = str(np.max(arr[3, :]))
            data_array[jj, 2] = file.split('_')[-1][:-4]
        
        np.savez(output_id + ".X.npz", X)
        np.savez(output_id + ".Y.npz", Y)
        np.savetxt(output_id + ".file_list.txt", input_files, fmt="%s")
        np.savetxt(output_id + ".station_info.txt", data_array, fmt="%s")


def load_tensor_data(data_id: str) -> tuple:
    """
    Load tensor data using the pattern in data_id

    Args:
        data_id (str): A file name pattern

    Returns:
        X (array): training data
        Y (array): training labels
        station_info (array): station information corresponding to the data in "X"
    """
    if os.path.isfile(data_id + "_train.X.npz"):
        print("Loading testing and training data from " + data_id)

        x_train = np.load(data_id + '_train.X.npz')['arr_0']
        y_train = np.load(data_id + '_train.Y.npz')['arr_0']

        x_test = np.load(data_id + '_test.X.npz')['arr_0']
        y_test = np.load(data_id + '_test.Y.npz')['arr_0']

        test_station_info = np.loadtxt(data_id + '_train.station_info.txt', dtype='str')
        train_station_info = np.loadtxt(data_id + '_train.station_info.txt', dtype='str')

        X = np.concatenate((x_train, x_test), axis=0)
        Y = np.concatenate((y_train, y_test), axis=0)
        station_info = np.concatenate((train_station_info, test_station_info), axis=0)

    elif os.path.isfile(data_id + ".X.npz"):
        print("Loading tensor data from " + data_id)
        X = np.load(data_id + '.X.npz')['arr_0']
        Y = np.load(data_id + '.Y.npz')['arr_0']
        station_info = np.loadtxt(data_id + '.station_info.txt', dtype='str')

    else:
        print("Error: Specified data not found. Bad data_id: " + data_id)
        X = None
        Y = None
        station_info = None
    
    return X, Y, station_info
