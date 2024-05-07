# © 2024. Triad National Security, LLC. All rights reserved.
"""
Build the tensors used in TensorFlow from .npy files
"""

from riddl.utils import data_io

if __name__ == '__main__':
    # Data directory and output ID
    data_dir = "./data/Bishop_etal2022"

    labels = ['transient', 'noise', 'moving', 'persistent']
    merge_labels = ['transient', 'moving']
    output_dir = "."
    output_id = "soi_example"
    test_fraction = 0.25
    file_pattern = None

    ###################
    # END USER INPUT  #
    ###################

    # Read in file list
    file_list, counts = data_io.build_file_list(data_dir, labels=labels, folders=labels, file_pattern=file_pattern)

    # Build tensors from data lists - data preparation functions
    output_id = output_dir + "/" + output_id
    data_io.write_tensor_data(file_list, output_id, labels=labels, test_fraction=test_fraction, merge_labels=merge_labels)