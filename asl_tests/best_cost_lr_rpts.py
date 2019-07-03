"""
Figure 1: Convergence of cost by learning rate
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)

plt.figure(figsize=(4, 4))
best_cost = []
for lr in LEARNING_RATES:
    subdir = "rpts_lr_%.3f" % lr
    cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
    best_cost.append(np.mean(cost_history, axis=(0, 1, 2))[-1])

plt.plot(best_cost)
plt.title("Best cost by learning rate")
plt.ylabel("Best cost achieved")
plt.xlabel("Learning rate")
#plt.ylim(35, 100)
plt.xticks(range(len(LEARNING_RATES)), ["%.2f" % lr for lr in LEARNING_RATES])
#plt.legend()

plt.savefig("best_cost_lr_asl_rpts.png")
plt.show()
