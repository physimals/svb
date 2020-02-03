"""
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

NT = 100
truth = nib.load("sim_data_biexp_%i_truth.nii.gz" % NT).get_data()
noisy = nib.load("sim_data_biexp_%i_noisy.nii.gz" % NT).get_data()

true_ts = truth[10, 0, 0, :]
noisy_ts = noisy[10, 0, 0, :]

plt.title("Sample timeseries (NT=%i)" % NT)
plt.plot(true_ts, 'g-')
plt.plot(noisy_ts, 'bx')

plt.savefig("sample_timeseries.png")
plt.show()
