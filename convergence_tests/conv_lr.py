"""
Figure 1: Convergence of cost by learning rate
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
#BATCH_SIZES = (5, 10, 15, 25, 50)
NT = (10, 20, 50, 100)
BS = 10

for idx, nt in enumerate(NT):
    plt.subplot(2, 2, idx+1)
    y0 = 1e10
    for lr in LEARNING_RATES:
        subdir = "nt_%i_lr_%.3f_bs_%i" % (nt, lr, BS)
        label = "LR: %.3f" % lr
        cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
        plt.plot(np.mean(cost_history, axis=(0, 1, 2)), label=label)
        if y0 > np.min(cost_history):
            y0 = np.min(cost_history)

    plt.title("Convergence of average free energy by learning rate (NT=%i)" % nt)
    #plt.yscale("symlog")
    plt.ylabel("Cost")
    plt.xlabel("Epochs")
    plt.ylim(y0, y0*10)
    plt.legend()

plt.show()
