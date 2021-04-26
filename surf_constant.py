

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

plds = 1.5 * np.ones(1)
repeats = 10
data = np.zeros((*ref_spc.size, plds.size * repeats))
mask = (pvs[...,:2] > 0.01).any(-1)

data_model = SurfaceModel(ref_spc.make_nifti(data), mask=mask, projector=projector)
asl_model = AslRestModel(data_model,
            plds=plds, repeats=repeats, casl=True)

CBF = [60, 20]
ATT = [1.3, 1.6]
NOISE_STD = 0

# idx = 15000
# lps = projector['LPS']
# neighbours = (lps.tris == idx).any(-1)
# neighbours = np.unique(lps.tris[neighbours,:])
# for _ in range(5):
#     neighbours = np.isin(lps.tris, neighbours)
#     neighbours = np.unique(lps.tris[neighbours.any(-1),:])

vox_cent = ref_spc.voxel_centres()
scale = 3
sine = (np.sin(vox_cent[...,0] / scale)
        + np.sin(vox_cent[...,1] / scale)
        + np.sin(vox_cent[...,2]) / scale)
sine = sine / sine.max()

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

ctx_cbf = 10 + (5 * ctx_sine)
LMS.save_metric(ctx_cbf[:LMS.n_points], 'L_sine.func.gii')
RMS.save_metric(ctx_cbf[-RMS.n_points:], 'R_sine.func.gii')

tpts = asl_model.tpts()
with tf.Session() as sess:

    # cbf = CBF[0] * np.ones([data_model.n_nodes, 1], dtype=np.float32)
    # cbf[neighbours] = 100
    # lps.save_metric(cbf[:lps.n_points], 'L_cortex_truth')
    att = ATT[0] * np.ones([data_model.n_nodes, 1], dtype=np.float32)

    hybrid_data = sess.run(asl_model.evaluate([
        ctx_cbf.astype(np.float32)
    ], tpts))
    hv_data = data_model.nodes_to_voxels(hybrid_data, True)
    data[mask,:] = sess.run(hv_data)

data += np.random.normal(0, NOISE_STD, data.shape)
data = data.reshape(*ref_spc.size, tpts.shape[-1])
if not os.path.exists('simdata.nii.gz') or True:
    ref_spc.save_image(data, 'simdata.nii.gz')


options = {
    "mode": "surface",
    "learning_rate" : 0.5,
    "batch_size" : 10,
    "sample_size" : 5,
    "epochs" : 700,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "projector" : projector,
    # "plds": plds, 
    # "repeats": repeats, 
    # "casl": True, 
    "prior_type": "M",
    "display_step": 10, 
    "outformat": ['gii'],
    "save_param_history": True, 

    "save_mean": True, 
    # "save_var": True, 
    # "save_std": True, 
    "save_cost": True, 
    # "save_cost_history": True, 
    # "save_param_history": True, 
    "save_model_fit": True, 
    "save_runtime": True, 
    "save_input": True, 
    "save_noise": True,

    'gamma_q1': 1.0,
    'gamma_q2': 10,

}


# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    ref_spc.make_nifti(data), "constant",
    "surf_out", 
    **options)


# %%
