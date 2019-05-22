"""
Figure 1: Comparison of prior/posterior
"""

import os
import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
BATCH_SIZES = (5, 10, 15, 25, 50)
NT = 20

subdirs = [d for d in glob.glob("prior*") if os.path.isdir(d)]
best_cost = []
for subdir in subdirs:
    cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
    best_cost.append(np.ravel(cost_history[..., -1]))
    print(subdir, np.mean(best_cost[-1]))

plt.boxplot(best_cost, labels=subdirs, showfliers=False)
plt.title("Best cost by prior/posterior options")
plt.ylabel("Best cost achieved")
plt.xlabel("Prior/initial posterior")
plt.ylim(40, 400)
#plt.yscale("symlog")
plt.legend()

plt.show()
