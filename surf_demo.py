

# %%
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import toblerone as tob 
import regtricks as rt
import trimesh
from svb.main import run
import pyvista as pv 
from pyvista import PlotterITK
from svb_models_asl import AslRestModel 
from svb.data import SurfaceModel


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
shift = 1.5 * (np.random.rand(*mesh.vertices.shape) - 0.5)
orig = (ref_spc.fov_size / 2)

out_surf = tob.Surface.manual((out_r * mesh.vertices) + orig + shift, mesh.faces)
in_surf = tob.Surface.manual((in_r * mesh.vertices) + orig + shift, mesh.faces)
mid_surf = tob.Surface.manual((mid_r * mesh.vertices) + orig + shift, mesh.faces)

print(mid_surf)
mid_edges = mid_surf.edges()
vertex_dist = np.linalg.norm(mid_edges, axis=-1).reshape(-1,3).mean()
print("Mean vertex spacing on midsurface {:.2f}mm".format(vertex_dist))


# %%
import pickle
import os 
hemi = tob.Hemisphere.manual(in_surf, out_surf, 'L')
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

from scipy import sparse
gp = mid_surf.mesh_laplacian(distance_weight=1)

def sym(a):
    return not ((np.abs(a - a.T) >= 1e-6).max())

def is_nsd(a):
    return not (sparse.linalg.eigs(a)[0] > 0).any()

assert sym(gp)
assert is_nsd(gp)

plds = np.arange(0.25, 1.75, 0.25)
repeats = 8
data = np.zeros(3*[sq_len] + [ repeats * plds.size ])
mask = vol_mask.reshape(data.shape[:3])
surf_model = SurfaceModel(data, surfaces={'LMS': mid_surf}, projector=projector, mask=mask)
asl_model = AslRestModel(surf_model, plds=plds, repeats=repeats, casl=True) 

CBF = 60 
ATT = 0.75
NOISE_VAR = 1

ftiss = CBF * np.ones([surf_model.n_nodes, 1], dtype=np.float32)
deltiss = ATT * np.ones([surf_model.n_nodes, 1], dtype=np.float32)
tpts = asl_model.tpts()

with tf.Session() as sess:
    surf_data = sess.run(asl_model.evaluate([ftiss, deltiss], tpts))

vol_data = projector.surf2vol(surf_data, pv_weight=True)
vol_data += np.random.normal(0, NOISE_VAR, vol_data.shape)
pvs = projector.flat_pvs()
vox_idx = np.argmax(pvs[:,0])
vol_data = vol_data.reshape(*ref_spc.size, tpts.shape[-1])

# Fit options common to both runs 
options = {
    "learning_rate" : 0.1,
    "batch_size" : plds.size,
    "sample_size" : 5,
    "epochs" : 1000,
    "log_stream" : sys.stdout,
    "prior_type": "N",
    "mask" : mask,
    "projector" : projector,
    "surfaces" : { 'LMS': mid_surf }, 
    "plds": plds, 
    "repeats": repeats, 
    "casl": True, 
    "ak": 1e-4, 
    "infer_ak": True 
}

# # Fit all parameters in N mode: no spatial prior, ie independent voxel fit 
# runtime_n, svb_n, training_history_n = run(
#     data_pv, "exp", 
#     "example_out_cov", 
#     **options)

# Fit amp1 and r1 in M mode: spatial prior 
runtime, svb, training_history = run(
    vol_data, "aslrest",
    "example_out_cov", 
    param_overrides={
        "ftiss" : { "prior_type": "M" },
        "delttiss" : { "prior_type": "M" }
    },
    **options)


# # %%
# amp_n, decay_n = svb_n.evaluate(svb_n.model_means)
# noise_n = svb_n.evaluate(svb_n.noise_mean)

# amp, decay = svb.evaluate(svb.model_means)
# noise = svb.evaluate(svb.noise_mean)
# amp_var, decay_var = svb.evaluate(svb.model_vars)


# # %%
# plot = PlotterITK()
# plot.add_mesh(mid_surf.to_polydata(), scalars=amp)
# plot.show()

# null.tpl [markdown]
# # Plot vertex- (model) and voxel-wise (noise) parameter means. N refers to non-spatial estimation, M is spatial. As expected, M reduces the variance in estimated parameter means for the model parameter, and has little effect on noise (because the only the *model* priors were set as M, not the noise itself). 

# # %%
# fig, axes = plt.subplots(1,3, constrained_layout=True)
# fig.set_size_inches(12,4)
# axes[0].set_title('Vertexwise amplitude')
# axes[0].boxplot([amp_n, amp], labels=['N', 'M'])
# axes[1].set_title('Vertexwise decay')
# axes[1].boxplot([decay_n, decay], labels=['N', 'M'])
# axes[2].set_title('Voxelwise noise SD')
# axes[2].boxplot([noise, noise_n], labels=['N', 'M'])
# plt.show()

# null.tpl [markdown]
# # **The following analysis is restricted to the M outputs only**
# # 
# # Plot vertex-wise parameter mean and variance against each other. There seems to be a small negative correlation in amplitude; no notable pattern in decay. 

# # %%
# fig, axes = plt.subplots(1,2, constrained_layout=True)
# fig.set_size_inches(8,4)
# axes[0].scatter(amp, amp_var)
# axes[1].scatter(decay, decay_var)
# for idx,title in enumerate(['Apmplitude', 'Decay']):
#     axes[idx].set_title(title)
#     axes[idx].set_xlabel('post mean')
#     axes[idx].set_ylabel('post var')

# null.tpl [markdown]
# # Plot vertexwise parameter means against each other. During the SVB run, the two model parameters were assumed to be independent of each other. A positive correlation is seen between amplitude and decay (why?). 

# # %%
# plt.scatter(amp, decay)
# plt.title('Vertexwise parameter means')
# plt.xlabel('Amplitude mean')
# plt.ylabel('Decay mean')
# plt.show()

# null.tpl [markdown]
# # Plot vertexwise parameter means against the PV of the voxel that the vertex associates to most strongly. This information is extracted from the projection matrix. Although a vertex can contribute to multiple voxels, we pick the one to which it contributes the most weight. One important aspect of surface SVB is that PVE are implicitly accounted for during the estimation process via the projection matrix. Hence, we should not see a corrrelation between voxel PV and estimated parameter means, and this does seem to be the case. 

# # %%
# maxvox_inds = projector.surf2vol_matrix().argmax(0).A.flatten()
# maxvox_pvs = projector.flat_pvs()[maxvox_inds,0]
# fig, axes = plt.subplots(1,2, constrained_layout=True)
# fig.set_size_inches(8,4)
# axes[0].scatter(amp, maxvox_pvs)
# axes[1].scatter(decay, maxvox_pvs)
# for idx,title in enumerate(['Apmplitude', 'Decay']):
#     axes[idx].set_title(title)
#     axes[idx].set_xlabel('post mean')
#     axes[idx].set_ylabel('max voxel PV')

# null.tpl [markdown]
# # Plot vertexwise parameter variances against the PV of the voxel that the vertex associates to most strongly. Based on the plot, it would seem that lower voxel PVs do not correlate with increased parameter uncertainty. 

# # %%
# maxvox_inds = projector.surf2vol_matrix().argmax(0).A.flatten()
# maxvox_pvs = projector.flat_pvs()[maxvox_inds,0]
# fig, axes = plt.subplots(1,2, constrained_layout=True)
# fig.set_size_inches(8,4)
# axes[0].scatter(amp_var, maxvox_pvs)
# axes[1].scatter(decay_var, maxvox_pvs)
# for idx,title in enumerate(['Apmplitude', 'Decay']):
#     axes[idx].set_title(title)
#     axes[idx].set_xlabel('post mean')
#     axes[idx].set_ylabel('post var')
# plt.show()

