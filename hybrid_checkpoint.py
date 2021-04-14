

# %%
import os
import sys 
import numpy as np
import toblerone as tob 
import regtricks as rt
import trimesh
from svb.main import run
from svb.data import SurfaceModel, VolumetricModel, HybridModel
import pickle

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

plds = np.arange(0.5, 1.75, 0.25)
repeats = 8
data = np.zeros((*ref_spc.size, plds.size * repeats))
mask = (pvs[...,:2] > 0.01).any(-1)

data_model = HybridModel(ref_spc.make_nifti(data), mask=mask, projector=projector)
asl_model = AslRestModel(data_model,
            plds=plds, repeats=repeats, casl=True)

CBF = [60, 20]
ATT = [1.3, 1.6]
NOISE_STD = 5

tpts = asl_model.tpts()
with tf.Session() as sess:
    cbf = CBF[0] * np.ones([data_model.n_nodes, 1], dtype=np.float32)
    cbf[data_model.vol_slicer] = CBF[1]
    # cbf[data_model.subcortical_slicer] = 10
    att = ATT[0] * np.ones([data_model.n_nodes, 1], dtype=np.float32)
    att[data_model.vol_slicer] = ATT[1]
    hybrid_data = sess.run(asl_model.evaluate([
        cbf, att
    ], tpts))
    hv_data = data_model.nodes_to_voxels(hybrid_data, True)
    data[mask,:] = sess.run(hv_data)

data += np.random.normal(0, NOISE_STD, data.shape)
data = data.reshape(*ref_spc.size, tpts.shape[-1])
if not os.path.exists('simdata.nii.gz') or True:
    ref_spc.save_image(data, 'simdata.nii.gz')


options = {
    "mode": "hybrid",
    "learning_rate" : 0.1,
    "batch_size" : plds.size,
    "sample_size" : 3,
    "epochs" : 1000,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "projector" : projector,
    "plds": plds, 
    "repeats": repeats, 
    "casl": True, 
    "prior_type": "M",
    "save_model_fit": True, 
    "display_step": 10, 
}


# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    ref_spc.make_nifti(data), "aslrest",
    "hybrid_out", 
    **options)


