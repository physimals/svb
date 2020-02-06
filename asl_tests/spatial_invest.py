"""
Run ASL tests
"""
import os
import sys

import numpy as np
import tensorflow as tf
import nibabel as nib

import fabber

import svb.main

# Test data properties
FNAME_RPTS = "mpld_asltc_diff.nii.gz"
FNAME_MEAN = "mpld_asltc_diff_mean.nii.gz"
FNAME_MASK = "mpld_asltc_mask.nii.gz"
PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SLICEDT = 0.0452
TAU = 1.8
RPTS = 8
CASL = True
BAT = 1.3
BATSD = 0.5

# Optimization properties
EPOCHS = 500

# Output options
BASEDIR = "/mnt/hgfs/win/data/svb/asl3"

kwargs = dict(
    plds=PLDS,
    slicedt=SLICEDT,
    repeats=[RPTS],
    tau=TAU,
    epochs=1,
    learning_rate=0.1,
    sample_size=1,
    infer_covar=False,
    force_num_latent_loss=True,
    param_overrides={
        "ftiss" : {
            "prior_type" : "M4",
        }
    },
)

tf.set_random_seed(1)
np.random.seed(1)

data_model = svb.DataModel(os.path.join(BASEDIR, FNAME_RPTS), os.path.join(BASEDIR, FNAME_MASK), **kwargs)
fwd_model = svb.get_model_class("aslrest")(**kwargs)
svb = svb.SvbFit(data_model, fwd_model, **kwargs)

tpts = fwd_model.tpts(data_model)
if tpts.ndim > 1 and tpts.shape[0] > 1:
    tpts = tpts[data_model.mask_flattened > 0]

svb.train(tpts, data_model.data_flattened, **kwargs)
ll = np.sum(svb.evaluate(svb.latent_loss))
print("ll", ll)
sample = svb.evaluate("post_sample")
print("sample", sample[:, 0, 0])
ak = svb.evaluate("ak")
print("ak", ak)
term1 = svb.evaluate("term1")
print("t1", np.sum(np.squeeze(term1)))
term2 = svb.evaluate("term2")
print("t2", np.mean(np.squeeze(term2)))