"""
Figure 1: Convergence of cost by learning rate
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)

for lr in LEARNING_RATES:
    subdir = "lr_%.3f_bs_10" % lr
    label = "%.3f" % lr
    cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
    plt.plot(np.mean(cost_history, axis=(0, 1, 2)), label=label)

plt.title("Convergence of VB-I-4 and VB-NI-4 analysis by average free energy")
plt.yscale("symlog")
plt.ylabel("Cost")
plt.xlabel("Epochs")
plt.ylim(35, 200)
plt.legend()

plt.show()
