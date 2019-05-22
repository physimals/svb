"""
Figure 1: Comparison of prior/posterior
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
BATCH_SIZES = (5, 10, 15, 25, 50)
NT = 20

epochs_converged = []
labels = []
for prior in ("i", "ni"):
    for post in ("ni", "ni_init", "i_true", "i_init", "i", "i_wrong"):
        subdir = "prior_%s_post_%s" % (prior, post)
        cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
        mean_cost_history = np.mean(cost_history, axis=(0, 1, 2))
        if mean_cost_history[-1] > 80:
            print("%s: skipping as mean cost not converged" % subdir)
            continue
        best_cost = cost_history[..., -1]
        non_converged = np.count_nonzero(best_cost > 80)
        converged = cost_history < np.expand_dims(best_cost, axis=-1)*1.05 # Within 5% of best cost
        epoch_converged = np.argmax(converged, axis=3)
        slow_converged = np.count_nonzero(epoch_converged > 200)
        print("%s: Non-converged voxels=%i, slow convergers=%i" % (subdir, non_converged, slow_converged))
        epochs_converged.append(epoch_converged[epoch_converged > 0])
        labels.append(subdir)

plt.boxplot(epochs_converged, labels=labels)
plt.title("Epoch converged by prior/posterior options")
plt.ylabel("Epoch at which convergence achieved")
plt.xlabel("Prior/initial posterior")
plt.ylim(0, 400)

plt.show()
