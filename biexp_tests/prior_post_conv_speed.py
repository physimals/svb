"""
Convergence speed by prior/posterior
"""

import os

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01)
BATCH_SIZES = (5, 10, 15, 25, 50)
NT = 20

BASEDIR="c:/Users/ctsu0221/dev/data/svb/biexp/"

readable_labels = {
#    "i_i" : "Informative\nDefault",
    "i_i_init" : "Informative\nData initialized",
    "i_i_true" : "Informative\nTrue",
    "i_i_wrong" : "Informative\nWrong",
#    "i_ni" : "Informative\nNon-informative",
#    "i_ni_init" : "Informative\nNon-informative\nData initialized",
#    "ni_i" : "Non-informative\nDefault",
    "ni_i_init" : "Non-informative\nData initialized",
    "ni_i_true" : "Non-informative\nTrue",
    "ni_i_wrong" : "Non-informative\nWrong",
#    "ni_ni" : "Non-informative\nNon-informative",
#    "ni_ni_init" : "Non-informative\nNon-informative\nData initialized",
}

epochs_converged = []
labels = []
plt.figure(figsize=(12, 8))
for prior in ("i", "ni"):
    for post in ("ni", "ni_init", "i_true", "i_init", "i", "i_wrong"):
        subdir = os.path.join(BASEDIR, "prior_%s_post_%s_analytic" % (prior, post))
        label = "%s_%s" % (prior, post)
        if not os.path.exists(os.path.join(subdir, "cost_history.nii.gz")):
            print("%s: skipping as not found" % subdir)
            continue
        if label not in readable_labels:
            print("%s: skipping as not included in label list" % subdir)
            continue

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
        labels.append(readable_labels[label])

plt.boxplot(epochs_converged, labels=labels)
plt.title("Epoch converged by prior/posterior options")
plt.ylabel("Epoch at which convergence achieved")
plt.xlabel("Prior/initial posterior")
plt.ylim(0, 400)

plt.tight_layout()
plt.savefig("prior_post_conv_speed.png")
plt.show()
