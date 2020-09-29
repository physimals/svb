# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython


# %%
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import toblerone as tob 
import regtricks as rt
import pyvista as pv 
import trimesh
from svb.main import run

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)

pv.set_plot_theme('document')


# %%
# Set the properties of the simulated data
num_times = 200
sq_len = 5
vox_size = 2 
true_params = {
    "amp1" : 10.0, # the true amplitude of the model
    "r1" : 3.0,    # the true decay rate
    "noise_sd" : 0.5,
}
true_var = true_params["noise_sd"]**2

# For time being, make sure we have a good range of samples based on the choice of R1
dt = 5.0/(true_params["r1"] * num_times)


# %%
# Generate test data and write to filenames
shape = (sq_len, sq_len, sq_len, num_times)
data = np.zeros(shape)
for t_idx in range(num_times):
    t = t_idx*dt
    data[..., t_idx] = true_params["amp1"]*math.exp(-true_params["r1"]*t)
data_noisy = data + np.random.normal(0, true_params["noise_sd"], size=shape)
data_noisy.shape


# %%
# Create a reference voxel grid for the data 
ref_spc = rt.ImageSpace.create_axis_aligned(np.zeros(3), 3 * [sq_len], 3 * [vox_size])

# Generate the point cloud of voxel centres
vox_cents = ref_spc.voxel_centres().reshape(-1,3)
voxel_poly = pv.PolyData(vox_cents)
ref_spc

# %%
# Generate a spherical surface
sph_mesh = trimesh.creation.icosphere(2, ((sq_len * vox_size)/2.2))
surf = tob.Surface.manual(sph_mesh.vertices + (sq_len * vox_size)/2,    
                                sph_mesh.faces)


# %%
# Form the weighting matrix for surface -> voxels
# We will use weighted averaging (no PV weighting etc)
surf2vol_weights = np.zeros((ref_spc.size.prod(), surf.points.shape[0]))
vertices_ijk = rt.aff_trans(ref_spc.world2vox, surf.points).round(0)
vertices_inds = np.ravel_multi_index(vertices_ijk.T.astype('i4'), ref_spc.size)
for vtx_number, vtx_ind in enumerate(vertices_inds):
    surf2vol_weights[vtx_ind, vtx_number] = 1 

assert (surf2vol_weights.sum(0) == 1).all()
divisor = surf2vol_weights.sum(1)
mask = (divisor > 0)
surf2vol_weights[surf2vol_weights == 0] = -1
surf2vol_weights[mask,:] /= divisor[mask,None]
surf2vol_weights[surf2vol_weights < 0] = 0
surf2vol_weights = surf2vol_weights[mask,:]
assert (surf2vol_weights.sum(1) == 1).all()


# %%
# Train model without covariance
options = {
    "learning_rate" : 0.005,
    "batch_size" : 10,
    "sample_size" : 8,
    "epochs" : 500,
    "log_stream" : sys.stdout,
    "n2v" : surf2vol_weights,
    "prior_type": "N",
    "param_overrides": {
        "amp1": { "prior_type": "M" } 
        } 
}


# %%
runtime, svb, training_history = run(
    data_noisy, "exp", 
    "example_out_cov", 
    mask=mask,
    surfaces={'LMS': surf}, 
    infer_covar=True, 
    dt=dt,
    **options)

mean_cost_history_cov = training_history["mean_cost"]
cost_history_v_cov = training_history["voxel_cost"]
param_history_v_cov = training_history["voxel_params"]
modelfit_cov = svb.evaluate(svb.modelfit)
means_cov = svb.evaluate(svb.model_means)
variances_cov = svb.evaluate(svb.model_vars)


# %%



