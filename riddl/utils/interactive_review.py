# © 2024. Triad National Security, LLC. All rights reserved.
"""
Methods to interactively review analysis windows, categorize,
and write .npy samples into directories for use later in 
machine learning algorithm training and evaluation.

"""

import os
import re

from importlib.util import find_spec
import numpy as np

import matplotlib.pyplot as plt

import pathos.multiprocessing as mp
from multiprocessing import cpu_count

from obspy.core import read, UTCDateTime
from infrapy.detection import beamforming_new

import subprocess

def instructions():
    temp = '\t' + "Enter category for analysis window:"
    temp += '\n\t\t' + "noise: n"
    temp += '\n\t\t' + "transient: t"
    temp += '\n\t\t' + "low-confidence transient: b"
    temp += '\n\t\t' + "persistent: p"
    temp += '\n\t\t' + "moving: m"
    temp += '\n\t\t' + "(any) aerial: a"
    temp += '\n\t\t' + "combination: c"
    temp += '\n\t\t' + "unclear: u"
    temp += '\n\t\t' + "pass: d"
    temp += '\n\t\t' + "quit: q"
    temp += '\n\tCategory: '

    return temp

if __name__ == '__main__':

    # Data information
    # data_path = find_spec('infrapy').submodule_search_locations[0][:-8] + "/infrapy-data/cli"
    # Raw Data Path
    # data_path = "/Users/jwbishop/Documents/Github/infrapy/infrapy-data/cli"
    data_path = "/Users/jwbishop/Documents/ML_New/DATA/"
    # Array Processed Data Path
    proc_path = "/Users/jwbishop/Documents/ML_New/Processed/arrays/"
    # Review Data Path
    rev_path = "/Users/jwbishop/Document/ML_New/Processed/"

    # array_ids = ["FSU", "HWU", "LCM", "PSU", "WMU"]
    # array_ids = ["I57_20191020", "I57_20191021"]
    array_ids = ["I53_20190927"]

    freq_min = 0.1
    freq_max = 5.0
    window_length = 20.0
    window_step = 5.0

    back_az_vals = np.arange(-180.0, 180.0, 2.0)
    trc_vel_vals = np.arange(300.0, 600.0, 2.5)
    cpu_cnt = cpu_count() - 1

    # Analysis window parameters (20 minutes for current work)
    ML_win_len = 20 * 60

    stop1 = False
    # Plotting flags
    slowness = True
    baz_adjust = True

    # Add loop here to loop through stations
    for array_id in array_ids:

        if stop1 is True:
            break
        else:
            pass

        # Prepare directories to hold results
        if not os.path.isdir(rev_path + "review3"):
            subprocess.call("mkdir " + rev_path + "review3", shell=True)

        for subdir in ["combo", "moving", "noise", "persistent", "transient", "unclear", "low_conf_transient", "aerial"]:
            if not os.path.isdir(rev_path + "review3" + "/" + subdir):
                subprocess.call("mkdir " + rev_path + "review3" + "/" + subdir, shell=True)
        
        # Read in data, set up array data
        # Change for IMS data
        st = read(data_path + array_id + "/" + array_id + ".SAC")
        st_info = st[0].stats.network + "." + st[0].stats.station + "." + st[0].stats.location + "." + st[0].stats.channel
        st.filter('bandpass', freqmin=freq_min, freqmax=freq_max)
        x, t, t0, geom = beamforming_new.stream_to_array_data(st)
        M, N = x.shape

        # If not os.path.isfile(array_id + ".beam_results.npy"):
        # Change for IMS data
        if not os.path.isfile(proc_path + array_id + "/" + array_id + "beam_results.npy"):
            print("Fiels not available! Run `run_analysis.py` for array processing first.")
            pass
        else:
            # Change for IMS data
            print("Reading previous results for " + array_id + "*.SAC beamforming analysis...")
            beam_times = np.load(proc_path + array_id + "/" + array_id + ".beam_times.npy")
            beam_results = np.load(proc_path + array_id + "/" + array_id + ".beam_results.npy")

        # Plot each window
        if slowness is True:
            fig, axis = plt.subplots(6, figsize=(10, 8), sharex=True)
        else:
            fig, axis = plt.subplots(4, figsize=(10, 8), sharex=True)
        axis[0].set_ylabel("F-value")
        axis[1].set_ylabel("Trace Velocity\n[m/s]")
        axis[2].set_ylabel("Back Azimuth\n[deg.]")
        axis[3].set_ylabel("Pressure\n[Pa]")
        if slowness is not True:
            axis[3].set_xlabel("Time (rel. " + str(t0) + ") [s]")
            axis[3].set_xlim([0, ML_win_len])
        else:
            axis[4].set_ylabel("S_x\n[s/m]")
            axis[5].set_ylabel("S_y\n[s/m]")
            axis[5].set_xlabel("Time (rel. " + str(t0) + ") [s]")
            axis[5].set_xlim([0, ML_win_len])
        
        print("Displaying windows for categorization...")
        for window_start in np.arange(0.0, (beam_times[-1] - beam_times[0]).astype(float) / 1.0e6, ML_win_len):
            st_mask = np.logical_and(window_start <= t, t <= window_start + ML_win_len)
            beam_mask = np.logical_and(window_start <= (beam_times - t0).astype(float) / 1.0e6, (beam_times - t0).astype(float) / 1.0e6 <= window_start + ML_win_len)
            print('\n' + "Categorizing signal at ", beam_times[beam_mask][0])

            if baz_adjust is True:
                beam_baz = (beam_results[:, 0][beam_mask] + 360) % 360
            else:
                beam_baz = beam_results[:, 0][beam_mask]
            
            ms = 4
            plot1 = axis[3].plot(t[st_mask] - window_start, x[0, :][st_mask], '-k')
            plot2 = axis[2].plot((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, beam_baz, '.k', markersize=ms)
            plot3 = axis[1].plot((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, beam_results[:, 1][beam_mask], '.k', markersize=ms)
            plot4 = axis[0].plot((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, beam_results[:, 2][beam_mask], '.k', markersize=ms)
            if slowness is True:
                s_mag = 1 / beam_results[:, 1][beam_mask]
                sx = np.cos(beam_results[:, 0][beam_mask] * np.pi / 180) * s_mag
                sy = np.sin(beam_results[:, 0][beam_mask] * np.pi / 180) * s_mag
                plot5 = axis[4].plot((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, sx, '.k', markersize=ms)
                plot6 = axis[5].plot((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, sy, '.k', markersize=ms)
            if baz_adjust is True:
                axis[2].set_ylim((0, 360))
            else:
                axis[2].set_ylim((-180, 180))
            # F-stat upper bound
            fstatmax = np.ceil(np.max(beam_results[:, 2][beam_mask]))
            if fstatmax < 2.0:
                fstatmax = 2.0
            axis[0].set_ylim([-0.5, fstatmax])
            axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0))
            plt.show(block=False)
            plt.pause(0.01)

            stop2 = False
            confirm_check = False
  
            while not confirm_check:
                category_check = False
                category = input(instructions())
                if 'n' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: noise [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> noise (y/n): "
                    file_path = rev_path + "review2/noise/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_noise"

                elif 't' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: transient signal [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> transient signal (y/n): "
                    file_path = rev_path + "review2/transient/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_transient"

                elif 'p' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: persistent signal [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> persistent (y/n): "
                    file_path = rev_path + "review2/persistent/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_persistent"

                elif 'm' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: moving source [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> moving source(y/n): "
                    file_path = rev_path + "review2/moving/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_moving"
                
                elif 'a' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: aerial [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> aerial (y/n): "
                    file_path = rev_path + "review2/aerial/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_aerial"

                elif 'c' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: combo [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> combo (y/n): "
                    file_path = rev_path + "review2/combo/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_combo"

                elif 'u' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: unclear [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> unclear(y/n): "
                    file_path = rev_path + "review2/unclear/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_unclear"

                elif 'b' in category:
                    axis[0].set_title(array_id + "-" + str(beam_times[beam_mask][0]) + "\n" + "Window Start = " + str(window_start/60.0) + '\n' + "Category: low-confidence transient [confirm (y/n)]")
                    category_check = True
                    confirm_text = "Confirm categorization --> low-confidence transient (y/n): "
                    file_path = rev_path + "review2/low_conf_transient/" + st_info + "." + re.sub("[:]", ".", str(beam_times[beam_mask][0])) + "_lct"
                
                elif 'd' in category:
                    # "d" for "do nothing..."
                    break

                elif 'q' in category:
                    category_check = True
                    confirm_text = "Confirm quit (y/n): "
                    stop2 = True
                
                else:
                    print('\t' + "Invalid categorization." + '\n')

                if category_check:
                    plt.pause(0.01)
                    confirm = input('\t' + confirm_text)
                    if 'y' in confirm and 'n' not in confirm:

                        if stop2 is True:
                            break

                        confirm_check = True

                        temp = np.vstack(((beam_times[beam_mask] - t0).astype(float) / 1.0e6 - window_start, beam_results[:, 0][beam_mask]))
                        temp = np.vstack((temp, beam_results[:, 1][beam_mask]))
                        temp = np.vstack((temp, beam_results[:, 2][beam_mask]))

                        np.save(file_path, temp)
                        print("Saved into " + file_path + ".npy")
                
            if stop2 is True:
                stop1 = True
                break

            plot1.pop(0).remove()
            plot2.pop(0).remove()
            plot3.pop(0).remove()
            plot4.pop(0).remove()
            if slowness is True:
                plot5.pop(0).remove()
                plot6.pop(0).remove()

        plt.clf()


            