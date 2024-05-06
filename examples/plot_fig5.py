#%%
import matplotlib.pyplot as plt
import numpy as np
from riddl.models import fk
import warnings
import obspy

from infrapy.utils import data_io

# Specify the ML model
model_id = "./models/use/soi2"
# Specify the fk_results from InfraPy
base_dir = "./data/Bishop_etal2022/Fig5/"
fk_label = base_dir + "fig5.fk_results.dat"
# Detection file
detection_file = base_dir + "fig5.p05.dets.json"
detection_file2 = base_dir + "fig5.p01.dets.json"
# SAC file
sac_file = base_dir + "I53H1.20191003_helicorder.SAC"
# Analyst picks
pick_file = base_dir + "analyst_review.txt"

#########################################
# Read files
#########################################
# Read the SAC file
st = obspy.read(sac_file)
st.filter("bandpass", freqmin=0.5, freqmax=5.0)
times = st[0].times('relative')
t_start = obspy.core.UTCDateTime('2019-10-03T01:00:00.00')

# Run ML analysis and summarize results
det_times, predictions = fk.run(fk_label, model_id)
for n in range(0, len(det_times)):
    print(det_times[n], '\t', predictions[n])

# Load in fk analysis results
print("Loading fk results from " + fk_label + "...")
try:
    fk_results = np.loadtxt(fk_label).T
except:
    warnings.warn("fk_results file not found")

# Load the adaptive F-detector detections
fk_detect_list = data_io.set_det_list(detection_file, merge=True)
fk_detections = []
print(fk_detect_list)
for det in fk_detect_list:
    print(det.peakF_UTCtime)
    fk_detections.append(det.peakF_UTCtime)
    t1 = det.peakF_UTCtime + np.timedelta64(int(det.start * 1000.0), 'ms')
    t2 = det.peakF_UTCtime + np.timedelta64(int(det.end * 1000.0), 'ms')

# Load the analyst picks
analyst_times = []
with open(pick_file) as pf:
    contents = pf.read()
    # print(contents)
    for line in contents.split('\n'):
        if len(line) > 0:
            if line[0] != '#':
                t = line.split(' ')[0]
                t = "2019-10-03T" + t
                analyst_times.append(obspy.core.UTCDateTime(t))

# Load the adaptive F-detector detections (p = 0.05)
fk_detect_list = data_io.set_det_list(detection_file, merge=True)
fk_detections = []
for det in fk_detect_list:
    fk_detections.append(det.peakF_UTCtime)
    t1 = det.peakF_UTCtime + np.timedelta64(int(det.start * 1000.0), 'ms')
    t2 = det.peakF_UTCtime + np.timedelta64(int(det.end * 1000.0), 'ms')

# Load the adaptive F-detector detections (p = 0.01)
fk_detect_list = data_io.set_det_list(detection_file2, merge=True)
fk_detections2 = []
for det in fk_detect_list:
    fk_detections2.append(det.peakF_UTCtime)
    t1 = det.peakF_UTCtime + np.timedelta64(int(det.start * 1000.0), 'ms')
    t2 = det.peakF_UTCtime + np.timedelta64(int(det.end * 1000.0), 'ms')

#%%
#########################################
# Plotting
#########################################
fig, axis = plt.subplots(4, 1, figsize=(15, 8), sharex=True)

# Adjust relative times
t0 = np.datetime64(t_start)
t0 = np.datetime64('2019-10-03T01:00:00.00')
plot_times = np.array([t0 + np.timedelta64(int(tn * 1000.0), 'ms') for tn in times])
plot_times2 = np.array([t0 + np.timedelta64(int(tn * 1000.0), 'ms') for tn in fk_results[0]])
detect_times = np.array([t_start + tn for tn in det_times])

# Sac file
axis[0].plot(plot_times, st[0].data, c='k')
axis[0].xaxis_date()
axis[0].set_ylabel("Pressure\n[Pa]", fontsize=12)
for jj in range(0, len(analyst_times)):
    axis[0].axvline(analyst_times[jj], linestyle='--', c='r')

# F-statistic
axis[1].scatter(plot_times2, fk_results[3], 0.25, c='k')
axis[1].set_ylabel("F statistic", fontsize=12)
axis[1].set_ylim(1.0, 7.0)

# Trace Velocity
axis[2].scatter(plot_times2, fk_results[2], 0.25, c='k')
axis[2].set_ylabel("Trace Velocity \n[m/s]", fontsize=12)
axis[2].set_ylim(300, 600)
axis[2].set_yticks([300, 400, 500, 600])
for jj in range(0, len(fk_detections)):
    axis[2].axvline(fk_detections[jj], linestyle='--', c='lime')
for jj in range(0, len(fk_detections2)):
    axis[2].axvline(fk_detections2[jj], linestyle=':', c='yellow')

# Back Azimuth
axis[3].scatter(plot_times2, fk_results[1], 0.25, c='k')
axis[3].set_ylabel("Back Azimuth \n[deg.]", fontsize=12)
axis[3].set_ylim(-180, 180)
axis[3].set_yticks([-180, -90, 0, 90, 180])
axis[3].set_xlim(plot_times[0], plot_times[-1])
axis[3].set_xlabel("Time [UTC] on Oct. 3rd, 2019", fontsize=12)
for jj in range(0, len(fk_detections)):
    axis[3].axvline(fk_detections[jj], linestyle='--', c='lime')
for jj in range(0, len(fk_detections2)):
    axis[3].axvline(fk_detections2[jj], linestyle=':', c='yellow')

# Add ML predictions
for n in range(0, len(detect_times)):
    # Signals of interest
    if predictions[n] == 1:
        for jj in range(1, 4):
            axis[jj].axvspan(detect_times[n] - 9.9 * 60, detect_times[n] + 9.9 * 60, color='blue', alpha=0.3)
    # Persistent signals
    elif predictions[n] == 2:
        for jj in range(1, 4):
            axis[jj].axvspan(detect_times[n] - 9.9 * 60, detect_times[n] + 9.9 * 60, color='green', alpha=0.3)

# %%
