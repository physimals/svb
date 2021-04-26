

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
from scipy import interpolate
from svb.models.misc import PolyModel

import sys 
import os.path as op
sys.path.append(op.join(op.dirname(__file__), '../svb_models_asl'))

from svb_models_asl import AslRestModel 

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)


projector = tob.Projector.load('proj.h5')
ref_spc = projector.spc 
pvs = projector.pvs() 
ref_spc.save_image(pvs, 'pvs.nii.gz')

plds = [1.5]
repeats = 50
data = np.zeros((*ref_spc.size, len(plds) * repeats))
mask = (pvs[...,:2] > 0.01).any(-1)

data_model = SurfaceModel(ref_spc.make_nifti(data), 
    mask=mask, projector=projector)
fwd_model = AslRestModel(data_model, plds=plds, repeats=repeats, casl=True)

inds = np.indices(projector.spc.size)
scale = 3
sine = np.sin(inds[1] / scale) + np.sin(inds[0] / scale) + np.sin(inds[2] / scale)
sine = sine / sine.max()
ref_spc.save_image(sine, 'sine_vol')

LMS, RMS = [ tob.Surface(p).transform(ref_spc.world2vox) 
            for p in [
                '103818.L.very_inflated.32k_fs_LR.surf.gii',
                '103818.R.very_inflated.32k_fs_LR.surf.gii'
            ] ]

ctx_sine = np.concatenate([
    interpolate.interpn(
        points=[ np.arange(d) for d in ref_spc.size ], 
        values=sine, 
        xi=s.points
    ) for s in [LMS, RMS]
])

cbf = 60 + (20 * np.maximum(ctx_sine, 0))
LMS.save_metric(cbf[:LMS.n_points], 'L_sine.func.gii')
RMS.save_metric(cbf[-RMS.n_points:], 'R_sine.func.gii')

tpts = fwd_model.tpts()
with tf.Session() as sess:

    hybrid_data = sess.run(fwd_model.evaluate([
        cbf[:,None].astype(np.float32)
    ], tpts))
    hv_data = data_model.nodes_to_voxels(hybrid_data, True)
    data[mask,:] = sess.run(hv_data)

SNR = 2
data += np.random.normal(0, data.max() / SNR, data.shape)
data = data.reshape(*ref_spc.size, tpts.shape[-1])
if not os.path.exists('simdata.nii.gz') or True:
    ref_spc.save_image(data, 'simdata.nii.gz')

data_surf = projector.vol2surf(data.mean(-1).flatten(), edge_scale=True)
LMS.save_metric(data_surf[:LMS.n_points], 'L_data_noisy.func.gii')
RMS.save_metric(data_surf[-RMS.n_points:], 'R_data_noisy.func.gii')


options = {
    "mode": "surface",
    "learning_rate" : 0.4,
    "batch_size" : 10,
    "sample_size" : 5,
    "epochs" : 250,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "projector" : projector,
    "casl": True, 
    "plds": plds, 
    "repeats": repeats, 
    "display_step": 10, 
    "outformat": ['gii'],
    "save_param_history": True, 

    "save_mean": True, 
    "save_cost": True, 
    "save_model_fit": True, 
    "save_runtime": True, 
    "save_input": True, 
    "save_noise": True,

    'gamma_q1': 1.0,
    'gamma_q2': 10,
    "prior_type": "M",

    # 'infer_ak': False, 
    # 'ak': 0.01

}


# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    ref_spc.make_nifti(data), "aslrest",
    "surf_out", 
    **options)


# %%
