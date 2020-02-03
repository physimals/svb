"""
Comparison best free energy by prior/posterior
"""

import os
import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

BASEDIR="c:/Users/ctsu0221/dev/data/svb/biexp/"

readable_labels = {
    "i_i" : "Inf/Default",
    "i_i_init" : "Inf/Data",
    "i_i_true" : "Inf/True",
    "i_i_wrong" : "Inf/Wrong",
    "i_ni" : "Inf/Noninf",
    "i_ni_init" : "Inf/Noninf-Data",
    "ni_i" : "Noninf/Default",
    "ni_i_init" : "Noninf/Data",
    "ni_i_true" : "Noninf/True",
    "ni_i_wrong" : "Noninf/Wrong",
    "ni_ni" : "Noninf/Noninf",
    "ni_ni_init" : "Noninf/Noninf-Data",
}

plt.figure(figsize=(10, 16))
ordered_labels = None
for idx, suffix in enumerate(("_analytic", "_num", "_analytic_corr", "_num_corr")):
    subdirs = [d for d in glob.glob(os.path.join(BASEDIR, "prior*")) if os.path.isdir(d) and d.endswith(suffix)]
    labels = [os.path.basename(s).replace("prior_", "").replace("post_", "").replace("_analytic", "").replace("_num", "").replace("_corr", "") for s in subdirs]
    cases = dict(zip(labels, subdirs))
    best_cost = []
    if ordered_labels is None:
        ordered_labels = [readable_labels[l] for l in sorted(cases.keys())]
        print(ordered_labels)
    for label in sorted(cases.keys()):
        subdir = cases[label]
        cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
        best_cost.append(np.ravel(cost_history[..., -1]))
        print(subdir, np.mean(best_cost[-1]))

    plt.subplot(4, 1, idx+1)
    plt.boxplot(best_cost, labels=ordered_labels, showfliers=False)
    plt.title("Best free energy by prior/posterior options: %s" % suffix)
    plt.ylabel("Best free energy achieved")
    plt.xlabel("Prior/initial posterior")
    plt.xticks(fontsize=6) 
    plt.ylim(0, 550)
    #plt.yscale("symlog")
    plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.savefig("prior_post.png")
plt.show()
