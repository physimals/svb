"""
Figure 1: Comparison of prior/posterior
"""

import os
import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

ordered_labels = None
for idx, suffix in enumerate(("_analytic", "_num", "_analytic_corr", "_num_corr")):
    subdirs = [d for d in glob.glob("prior*") if os.path.isdir(d) and d.endswith(suffix)]
    labels = [s.replace("prior_", "").replace("post_", "").replace("_analytic", "").replace("_num", "").replace("_corr", "") for s in subdirs]
    cases = dict(zip(labels, subdirs))
    best_cost = []
    if ordered_labels is None:
        ordered_labels = sorted(cases.keys())
        print(ordered_labels)
    for label in sorted(cases.keys()):
        subdir = cases[label]
        cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
        best_cost.append(np.ravel(cost_history[..., -1]))
        print(subdir, np.mean(best_cost[-1]))

    plt.subplot(2, 2, idx+1)
    plt.boxplot(best_cost, labels=ordered_labels, showfliers=False)
    plt.title("Best cost by prior/posterior options: %s" % suffix)
    plt.ylabel("Best cost achieved")
    plt.xlabel("Prior/initial posterior")
    plt.ylim(40, 400)
    #plt.yscale("symlog")
    plt.legend()

plt.show()
