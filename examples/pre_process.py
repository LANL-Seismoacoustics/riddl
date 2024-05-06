#%%
import numpy as np

from riddl.utils import data_io

if __name__ == '__main__':
    input_dir = "."
    file = "infrapy_example_long.fk_results.dat"

    labels = ['transient', 'noise', 'noise']
    output_dir = './data/fk_array_data'

    # This is the array length for the 
    # Do not change this value.
    data_length = 241

    beam_file = input_dir + "/" + file
    F_stat, baz, trace_vel = data_io.load_raw_fk_results(beam_file)

    # Determine the number of output files
    num_out_files = int(np.floor(len(trace_vel) / data_length))
    print("Creating " + str(num_out_files) + " .npy files.")

    # Loop through data and output labelled files
    start_counter = 0
    end_counter = data_length
    for jj in range(0, num_out_files):
        print("Using label '" + labels[jj] + "' for window " + str(jj) + ".")
        data = np.vstack((F_stat[start_counter:end_counter], trace_vel[start_counter:end_counter], baz[start_counter:end_counter]))
        start_counter += data_length
        end_counter += data_length
        file_name = file.split(".")[0] + "." + str(jj) + "." + labels[jj] + ".npy"
        np.save(output_dir + "/" + file_name, data)
