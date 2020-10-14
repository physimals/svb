# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # SVB surface test script
# 
# This script demonstrates SVB running 'on the surface' to infer the parameters of a single exponential decay signal model. 
# 
# Voxel-wise data is generated with a given amplitude, decay rate and noise SD. Each voxel in the volume is an independent time series of samples. Although they all share the same underlying signal parameters, the additive zero-mean (aka white) noise present in each one will be different. 
# 
# A spherical surface is then generated to intersect the voxel grid (note that many voxels will not intersect the surface and are therefore discarded). Multiple surface vertices may be present in each voxel, leading to an under-determined system. The SVB framework is used to infer the signal parameters at each vertex *on the surface* using the data generated in *volume space*. The process is run both with and without the spatial prior to illustrate how this is helpful in an under-determined system. 

# %%
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import toblerone as tob 
import regtricks as rt
import trimesh
from svb.main import run
import pickle

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)

# %% [markdown]
# Generate a cubic voxel grid of data. The signal parameters for the exponential model are amplitude and decay rate, and the noise is zero-mean with a given SD. Multiple samples are taken from the exponential curve to produce a 4D dataset (X,Y,Z,t)

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
dt = 5.0/(true_params["r1"] * num_times)

shape = (sq_len, sq_len, sq_len, num_times)
data = np.zeros(shape)
for t_idx in range(num_times):
    t = t_idx*dt
    data[..., t_idx] = true_params["amp1"]*math.exp(-true_params["r1"]*t)
data_noisy_nopv = data + np.random.normal(0, true_params["noise_sd"], size=shape)
print('Data shape', data_noisy_nopv.shape)

# %% [markdown]
# Create a reference voxel grid for the data 
# 

# %%
ref_spc = rt.ImageSpace.create_axis_aligned(np.zeros(3), 3 * [sq_len], 3 * [vox_size])
ref_spc

# %% [markdown]
# Generate a spherical surface, with radius slightly smaller than 1/2 the FoV of the reference space
# 

# %%
ctx_thickness = 1
mid_r = (sq_len * vox_size)/2.3
in_r = mid_r - ctx_thickness/2
out_r = mid_r + ctx_thickness/2
mesh = trimesh.creation.icosphere(2, 1) 
in_surf, mid_surf, out_surf = [
    tob.Surface.manual((r * mesh.vertices) + (sq_len * vox_size)/2, mesh.faces) for r in [in_r, mid_r, out_r]
]
print(in_surf)
print(mid_surf)
print(out_surf)

# %% [markdown]
# Calculate the mean vertex spacing on the mesh. As the Laplacian is a function of the second derivative, we may expect the smoothing parameter ak to vary in proportion to (1/length scale)^2?

# %%
mid_edges = mid_surf.edges()
vertex_dist = np.linalg.norm(mid_edges, axis=-1).reshape(-1,3).mean()
print("Mean vertex spacing on midsurface {:.2f}mm".format(vertex_dist))

# %% [markdown]
# Form the weighting matrix for surface to volume projection. The projection incorporates PV effects, hence why the input data must be scaled with PVs. 

# %%
hemi = tob.Hemisphere.manual(in_surf, out_surf, 'L')
if os.path.exists('proj.pkl'):
    with open('proj.pkl', 'rb') as f: 
        projector = pickle.load(f)
else: 
    projector = tob.projection.Projector(hemi, ref_spc)
    with open('proj.pkl', 'wb') as f: 
        pickle.dump(projector, f)
projector.surf2vol_matrix().shape

# %% [markdown]
# Mask out voxels with no vertices within. A cortical surface at 32k resolution in a voxel grid of ~3mm iso has around 10 vertices per voxel. 

# %%
vertices_per_voxel = (projector.surf2vol_matrix() > 0).sum(1).A.flatten()
vol_mask = (vertices_per_voxel > 0)
surf2vol_weights = projector.surf2vol_matrix()[vol_mask,:]
print("Mean vertices per voxel:", (surf2vol_weights > 0).sum(1).mean())


# %%
pvs = projector.flat_pvs()
data_noisy = data_noisy_nopv * pvs[:,0].reshape(ref_spc.size)[...,None]
# plt.figure(figsize=(8,4))
# plt.subplot(1,2,1)
# plt.title('Noisy data, no PVE')
# plt.imshow(data_noisy_nopv[:,:,5,0], vmin=data_noisy_nopv.min(), vmax=data_noisy_nopv.max())
# plt.subplot(1,2,2)
# plt.title('Noisy data, with PVE')
# plt.imshow(data_noisy[:,:,5,0], vmin=data_noisy_nopv.min(), vmax=data_noisy_nopv.max())
# plt.show()

# %% [markdown]
# Plot the surface to volume projection matrix

# %%
# plt.figure(figsize=(5,5))
# plt.spy(surf2vol_weights, markersize=0.5)
# plt.title('Surface to volume projection matrix')
# plt.xlabel('Vertex number')
# plt.ylabel('Voxel index (masked)')
# plt.show()


# %% [markdown]
# Repeat the inference process, this time with the spatial prior enabled for amplitude and decay. This enforces similarity between the estimates of neighbouring vertices. 

# %%
options = {
    "learning_rate" : 0.005,
    "batch_size" : 10,
    "sample_size" : 5,
    "epochs" : 1000,
    "log_stream" : sys.stdout,
    "n2v" : surf2vol_weights,
    "prior_type": "N",
    "ak": 30
}

runtime, svb, training_history = run(
    data_noisy, "exp", 
    "example_out_cov", 
    mask=vol_mask,
    surfaces={'LMS': mid_surf}, 
    dt=dt,
    **options)

mean_cost_history = training_history["mean_cost"]
cost_history_v = training_history["voxel_cost"]
param_history_v = training_history["model_params"]
modelfit = svb.evaluate(svb.modelfit)
means = svb.evaluate(svb.model_means)
variances = svb.evaluate(svb.model_vars)

# %% [markdown]
# For each parameter of interest, we infer a mean (best estimate) and variance (degree of uncertainty). We also project the surface parameter estimates back into volume space for reference. 

# %%
amp1_mean, r1_mean = [ *means ]
amp1_var, r1_var =  [ *variances ] 
noise_var = training_history["mean_noise_params"]
vol_amp1 = surf2vol_weights.dot(amp1_mean)

# %% [markdown]



# %%



