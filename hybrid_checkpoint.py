

# %%
import os
import sys 
import numpy as np
import toblerone as tob 
import regtricks as rt
import trimesh
from svb.main import run
from svb.data import SurfaceModel, VolumetricModel
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


sq_len = 10 
vox_size = 2
ref_spc = rt.ImageSpace.create_axis_aligned(np.zeros(3), 3 * [sq_len], 3 * [vox_size])
bounding_box = np.array([ref_spc.bbox_origin, ref_spc.bbox_origin + ref_spc.fov_size])
bounding_box = np.meshgrid(*[ se for se in bounding_box.T ])
ref_spc


ctx_thickness = 1
out_r = ((sq_len * vox_size) - 1) / 2
in_r = out_r - ctx_thickness
mid_r = out_r - (ctx_thickness / 2)
mesh = trimesh.creation.icosphere(2, 1)
orig = (ref_spc.fov_size / 2)

out_surf = tob.Surface.manual((out_r * mesh.vertices) + orig, mesh.faces)
in_surf = tob.Surface.manual((in_r * mesh.vertices) + orig, mesh.faces)
mid_surf = tob.Surface.manual((mid_r * mesh.vertices) + orig, mesh.faces)

print(mid_surf)
mid_edges = mid_surf.edges()
vertex_dist = np.linalg.norm(mid_edges, axis=-1).reshape(-1,3).mean()
print("Mean vertex spacing on midsurface {:.2f}mm".format(vertex_dist))


# %%
hemi = tob.Hemisphere(in_surf, out_surf, 'L')
if not os.path.exists('proj.pkl'):
    projector = tob.projection.Projector(hemi, ref_spc)
    with open('proj.pkl', 'wb') as f:
        pickle.dump(projector, f)
else: 
    with open('proj.pkl', 'rb') as f:
        projector = pickle.load(f)

# %%
vertices_per_voxel = (projector.surf2vol_matrix(True) > 0).sum(1).A.flatten()
vol_mask = (vertices_per_voxel > 0)
surf2vol_weights = projector.surf2vol_matrix(True)[vol_mask,:]
print("Mean vertices per voxel:", (surf2vol_weights > 0).sum(1).mean())


plds = np.arange(0.5, 1.75, 0.25)
repeats = 8
data = np.zeros(3*[sq_len] + [ repeats * plds.size ])
mask = vol_mask.reshape(data.shape[:3])
pvs = projector.flat_pvs().reshape(*ref_spc.size,-1)[...,:2]
# mask = np.ones_like(mask, dtype=np.bool)

# cortex_model = AslRestModel(
#         SurfaceModel(ref_spc.make_nifti(data), projector=projector), 
#         plds=plds, repeats=repeats, casl=True) 
subcortex_model = AslRestModel(
            VolumetricModel(data), plds=plds, repeats=repeats, casl=True,
            pvcorr=True, pvgm=pvs[...,0], pvwm=pvs[...,1])

CBF = [60, 20]
ATT = [1.3, 1.6]
NOISE_STD = 1
init_mean = np.concatenate([
    60 * np.ones(projector.n_surf_points),
    20 * np.ones(mask.sum().astype(np.int32))
])

tpts = subcortex_model.tpts()
with tf.Session() as sess:
    ones = np.ones([mask.size, 1], dtype=np.float32)
    subcortex_data = sess.run(subcortex_model.evaluate([
        CBF[0] * ones, ATT[0] * ones,
        CBF[1] * ones, ATT[1] * ones
    ], tpts))

vol_data = subcortex_data
vol_data += np.random.normal(0, NOISE_STD, vol_data.shape)
vol_data = vol_data.reshape(*ref_spc.size, tpts.shape[-1])
if not os.path.exists('simdata.nii.gz'):
    ref_spc.save_image(vol_data, 'simdata.nii.gz')


options = {
    "mode": "hybrid",
    "learning_rate" : 0.1,
    "batch_size" : plds.size,
    "sample_size" : 5,
    "epochs" : 1000,
    "log_stream" : sys.stdout,
    "mask" : mask,
    "projector" : projector,
    "plds": plds, 
    "repeats": repeats, 
    "casl": True, 
    "prior_type": "M",
    "save_model_fit": True, 
}


# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    ref_spc.make_nifti(vol_data), "aslrest",
    "hybrid_out", 
    **options)


