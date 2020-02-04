"""
Comparison best free energy by prior/posterior
"""

import os
import glob

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

BASEDIR="c:/Users/ctsu0221/dev/data/svb/biexpn/"

readable_labels = {
    "i_i" : "Informative\nDefault",
    "i_i_init" : "Informative\nData initialized",
    "i_i_true" : "Informative\nTrue",
    "i_i_wrong" : "Informative\nWrong",
    "i_ni" : "Informative\nNon-informative",
    "i_ni_init" : "Informative\nNon-informative\nData initialized",
    "ni_i" : "Non-informative\nDefault",
    "ni_i_init" : "Non-informative\nData initialized",
    "ni_i_true" : "Non-informative\nTrue",
    "ni_i_wrong" : "Non-informative\nWrong",
    "ni_ni" : "Non-informative\nNon-informative",
    "ni_ni_init" : "Non-informative\nNon-informative\nData initialized",
}

plt.figure(figsize=(10, 16))
ordered_labels = None
#for idx, suffix in enumerate(("_analytic", "_num", "_analytic_corr", "_num_corr")):
for idx, suffix in enumerate(("_analytic_corr",)):
    subdirs = [d for d in glob.glob(os.path.join(BASEDIR, "prior*")) if os.path.isdir(d) and d.endswith(suffix)]
    labels = [os.path.basename(s).replace("prior_", "").replace("post_", "").replace("_analytic", "").replace("_num", "").replace("_corr", "") for s in subdirs]
    cases = dict(zip(labels, subdirs))
    best_cost = []
    if ordered_labels is None:
        ordered_labels = [readable_labels[l] for l in sorted(cases.keys())]
        for l in ordered_labels:
            print(l)
        print(ordered_labels)
    for label in sorted(cases.keys()):
        subdir = cases[label]
        cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
        best_cost.append(np.ravel(cost_history[..., -1]))
        print(subdir, np.mean(best_cost[-1]))

    plt.subplot(4, 1, idx+1)
    plt.boxplot(best_cost, labels=ordered_labels, showfliers=False)
    plt.title("Best free energy by prior/posterior options")
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
