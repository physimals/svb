

# %%
import os
import os.path as op 
import sys 
import numpy as np
import matplotlib.pyplot as plt
import toblerone as tob 
import regtricks as rt
import trimesh
from svb.main import run
import pyvista as pv 
from pyvista import PlotterITK
from svb.data import SurfaceModel, VolumetricModel

import sys 
import os.path as op
sys.path.append(op.join(op.dirname(__file__), '../svb_models_asl'))

from svb_models_asl import AslRestModel 

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

def simulate_data(params, opts, shape):
    spc = rt.ImageSpace.create_axis_aligned([0,0,0], shape, [1,1,1])
    data = np.zeros((*spc.size, opts['repeats'] * len(opts['plds'])))
    asl_model = AslRestModel(
            VolumetricModel(spc.make_nifti(data)), **opts)

    tpts = asl_model.tpts()
    with tf.Session() as sess:
        ones = np.ones([asl_model.data_model.n_nodes, 1], dtype=np.float32)
        data = sess.run(asl_model.evaluate(
            [ p * ones for p in params], tpts))

    return data.reshape(*spc.size, tpts.shape[-1])


# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)

projector = tob.Projector.load('proj_27_24.h5')
spc = projector.spc 
pvs = projector.flat_pvs().reshape(*spc.size, -1)[...,:2]
mask = (pvs > 0.02).any(-1)
spc.save_image(pvs, 'pvs.nii.gz')

CBF = [60, 20]
ATT = [1.3, 1.6]
TAU = 1.4
REPEATS = 5 
plds = [0.8]
opts = dict(plds=plds, repeats=5, tau=TAU, casl=True, pvcorr=True, 
            mask=mask, pvgm=pvs[...,0], pvwm=pvs[...,1])
data = simulate_data([ CBF[0], CBF[1] ], opts, spc.size)



options = {
    "mode": 'hybrid', "projector": projector,
    # "outformat": ['gii', 'flatnii', 'nii'],

    "plds": plds, 
    "repeats": REPEATS, 
    "tau": TAU,
    "casl": True, 

    "learning_rate" : 0.05,
    "batch_size" : len(plds),
    "sample_size" : 5,
    "epochs" : 1000,
    "display_step": 20,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "ak": 5,
    "infer_ak": False,

    "save_mean": True, 
    "save_var": True, 
    "save_std": True, 
    "save_cost": True, 
    "save_cost_history": True, 
    "save_param_history": True, 
    "save_model_fit": True, 
    "save_post": True, 
    "save_runtime": True, 
    "save_input_data": True, 
    "save_noise": True
}

# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    spc.make_nifti(data), "aslrest",
    "demo_out", 
    param_overrides={
        "ftiss" : { "prior_type": "M" },
        # "fwm" : { "prior_type": "M" }
    },
    **options)


