import sys
import logging
import math

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

from svb import SvbFit, dist, DataModel
from svb.models.exp import BiExpModel
from svb.main import setup_logging

log = logging.getLogger(__name__)
setup_logging(tee=sys.stdout)

# Set the properties of the simulated data
num_times=200
num_examples = 9
name = "example_data"
true_params = {
    "amp1" : 10.0, # the true amplitude of the model
    "amp2" : 10.0,
    "r1" : 1.0,    # the true decay rate
    "r2" : 10.0,
    "noise_sd" : 1.0,
}
true_var = true_params["noise_sd"]**2

# For time being, make sure we have a good range of samples based on the choice of R1
dt = 5.0/(true_params["r1"] * num_times)

# Generate test data and write to filenames
sq_len = int(math.sqrt(num_examples))
shape = (sq_len, sq_len, 1, num_times)
data = np.zeros(shape)
for t_idx in range(num_times):
    t = t_idx*dt
    data[..., t_idx] = true_params["amp1"]*math.exp(-true_params["r1"]*t) + \
                       true_params["amp2"]*math.exp(-true_params["r2"]*t)
nii = nib.Nifti1Image(data, np.identity(4))
nii.to_filename(name + ".nii.gz")
data_noisy = data + np.random.normal(0, true_params["noise_sd"], size=shape)
nii_noisy = nib.Nifti1Image(data_noisy, np.identity(4))
nii_noisy.to_filename(name + "_noisy.nii.gz")

# Create the forward model
fwd_model = BiExpModel(dt=dt)
log.info("Created model: %s", str(fwd_model))
for option in fwd_model.OPTIONS:
    log.info(" - %s: %s", option.desc, str(getattr(fwd_model, option.attr_name)))

# Initialize the data model which contains data dimensions, number of time
# points, list of unmasked voxels, etc
data_model = DataModel(name + "_noisy.nii.gz")
data_model.nifti_image(data_model.data_flattened).to_filename("example_data.nii.gz")

# Get the time points from the model
tpts = fwd_model.tpts(data_model)
if tpts.ndim > 1 and tpts.shape[0] > 1:
    tpts = tpts[data_model.mask_flattened > 0]

# Train model without covariance
options = {
    "learning_rate" : 0.005,
    "batch_size" : 10,
    "sample_size" : 10,
    "epochs" : 500,
}
svb = SvbFit(data_model, fwd_model, **options)
log.info("Training model...")
ret = svb.train(tpts, data_model.data_flattened, **options)

# Get output, transposing as required so first index is by parameter
mean_cost_history = ret[0]
cost_history_v = ret[2]
param_history_v = ret[3]
modelfit = ret[4]
means = svb.evaluate(svb.model_means)
variances = np.transpose(svb.evaluate(svb.post.var))

# Train model with covariance
svb = SvbFit(data_model, fwd_model, infer_covar=True, **options)
log.info("Training model (with covariance)...")
ret = svb.train(tpts, data_model.data_flattened, **options)

# Get output, transposing as required so first index is by parameter
mean_cost_history_cov = ret[0]
cost_history_v_cov = ret[2]
param_history_v_cov = ret[3]
modelfit_cov = ret[4]
means_cov = svb.evaluate(svb.model_means)
variances_cov = np.transpose(svb.evaluate(svb.post.var))

# Write out parameter mean and variance images plus param history
for idx, param in enumerate(svb.params):
    data_model.nifti_image(means[idx]).to_filename("mean_%s.nii.gz" % param.name)
    data_model.nifti_image(variances[idx]).to_filename("var_%s.nii.gz" % param.name)
    data_model.nifti_image(param_history_v[:, :, idx]).to_filename("mean_%s_history.nii.gz" % param.name)
    
# Write out voxelwise cost history
data_model.nifti_image(cost_history_v[..., -1]).to_filename("cost.nii.gz")
data_model.nifti_image(cost_history_v).to_filename("cost_history.nii.gz")

# Write out modelfit
data_model.nifti_image(modelfit).to_filename("modelfit.nii.gz")

if "--show" in sys.argv:
    # Plot some results
    clean_data = data.reshape((-1, num_times))

    for idx in range(num_examples):
        plt.figure(1)
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(cost_history_v[idx])

        plt.figure(2)
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(cost_history_v_cov[idx])

        plt.figure(3)
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(tpts, data_model.data_flattened[idx],'rx')
        plt.plot(tpts, clean_data[idx], 'r')
        plt.plot(tpts, modelfit[idx],'b')

        plt.figure(4)
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(tpts, data_model.data_flattened[idx],'rx')
        plt.plot(tpts, clean_data[idx], 'r')
        plt.plot(tpts, modelfit_cov[idx],'b')

    plt.show()
