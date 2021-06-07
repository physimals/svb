import os
import sys 
import numpy as np
import toblerone as tob 
from svb.main import run
from svb.data import HybridModel
from scipy import interpolate

# import sys 
# import os.path as op
# sys.path.append(op.join(op.dirname(__file__), '../svb_models_asl'))
from svb_models_asl import AslRestModel 

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)


projector = tob.Projector.load('103818_L_hemi.h5')
ref_spc = projector.spc 
LMS = tob.Surface('103818.L.very_inflated.32k_fs_LR.surf.gii')
LMS = LMS.transform(ref_spc.world2vox)

plds = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
repeats = 8
data = np.zeros((*ref_spc.size, len(plds) * repeats))
mask = (projector.pvs()[...,:2] > 0.01).any(-1)

data_model = HybridModel(ref_spc.make_nifti(data), 
                   projector=projector)
asl_model = AslRestModel(data_model,
            plds=plds, repeats=repeats, casl=True)

ATT = [1.3, 1.6]
SNR = 10
N_VAR = 42 * np.sqrt(len(plds) * repeats) / SNR 

inds = np.indices(projector.spc.size)
scale = 3
sine = (np.sin(inds[1] / scale) 
        + np.sin(inds[0] / scale) 
        + np.sin(inds[2] / scale))
sine = sine / sine.max()

ctx_sine = interpolate.interpn(
        points=[ np.arange(d) for d in ref_spc.size ], 
        values=sine, 
        xi=LMS.points
    )

ctx_cbf = 60 + (25 * ctx_sine)
ctx_att = 1.3 * np.ones_like(ctx_cbf)
LMS.save_metric(ctx_cbf, 'L_ctx_sine.func.gii')

tpts = asl_model.tpts()
nvox = ref_spc.size.prod()

with tf.Session() as sess:

    cbf = np.concatenate([
            ctx_cbf[:,None],
            20 * np.ones([nvox, 1]), 
    ])
    att = np.concatenate([
            ctx_att[:,None],
            1.6 * np.ones([nvox, 1]), 
    ])
    data = sess.run(asl_model.evaluate(
            [ cbf.astype(np.float32), att.astype(np.float32) ], tpts))

data = projector.node2vol(data, edge_scale=True).reshape(*ref_spc.size, -1)
data[mask,:] += np.random.normal(0, N_VAR, size=data[mask,:].shape)
ref_spc.save_image(data, 'hybrid_simdata.nii.gz')

data_surf = projector.vol2surf(data.mean(-1).flatten(), edge_scale=False)
LMS.save_metric(data_surf, 'hybrid_simdata_mean_proj.func.gii')

options = {
    "mode": "hybrid",
    "learning_rate" : 0.5,
    "batch_size" : len(plds),
    "sample_size" : 5,
    "epochs" : 500,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "projector" : projector,
    "plds": plds, 
    "repeats": repeats, 
    "casl": True, 
    "prior_type": "N",
    "save_model_fit": True, 
    "display_step": 10, 
    "save_param_history": True, 
    "save_cost": True, 
    "save_cost_history": True, 
    "save_var": True,

    'gamma_q1': 1.0, 
    'gamma_q2': 100, 

}


# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    ref_spc.make_nifti(data), "aslrest",
    "hybrid_out", 
    **options)


