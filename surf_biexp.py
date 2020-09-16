# Example fitting of biexponential model
#
# This example runs SVB on a set of instances of biexponential data
# (by default just 4, to make it possible to display the results
# graphically, but you can up this number and not plot the output
# if you want a bigger data set)
#
# Usage: python biexp_exam
import sys
import math

import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf
import matplotlib.pyplot as plt
import nibabel as nib
import toblerone as tob 


# Use the package from the local directory, not the system installed version 
import os.path as op 
sys.path.append(op.abspath(op.join(__file__, '..', '..')))
sys.argv.append('--show')

from svb.main import run

# To make tests repeatable
tf.set_random_seed(1)
np.random.seed(1)

# Set the properties of the simulated data
num_times=200
num_examples = 4
name = "surf_test"
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

# Generate dummy surface 
points = np.array([
    [0,0,0], [1,0,0], [0,1,0], [1,1,0]
], dtype=np.float32)
tris = np.array([
    [0,3,1], [2,3,0]
], dtype=np.int32)
surfaces = { 'LMS': tob.Surface.manual(points, tris) }

# Train model without covariance
options = {
    "learning_rate" : 0.005,
    "batch_size" : 10,
    "sample_size" : 10,
    "epochs" : 250,
    "log_stream" : sys.stdout,
    "save_mean" : True,
    "save_var" : True,
    "save_param_history" : True,
    "save_cost" : True,
    "save_cost_history" : True,
    "save_model_fit" : True,
    "save_log" : True,
}

# runtime, svb, training_history = run(data_noisy, "biexp", "example_out_nocov", **options)
# mean_cost_history = training_history["mean_cost"]
# cost_history_v = training_history["voxel_cost"]
# param_history_v = training_history["voxel_params"]
# modelfit = svb.evaluate(svb.modelfit)
# means = svb.evaluate(svb.model_means)
# variances = svb.evaluate(svb.model_vars)
# post_mean = svb.evaluate(svb.post.mean)
# post_cov = svb.evaluate(svb.post.cov)

runtime, svb, training_history = run(data_noisy, "biexp", "surf_test_cov", None, surfaces, infer_covar=True, **options)
mean_cost_history_cov = training_history["mean_cost"]
cost_history_v_cov = training_history["voxel_cost"]
param_history_v_cov = training_history["voxel_params"]
modelfit_cov = svb.evaluate(svb.modelfit)
means_cov = svb.evaluate(svb.model_means)
variances_cov = svb.evaluate(svb.model_vars)

# runtime, svb, training_history = run(data_noisy, "biexp", "example_out_cov_init", infer_covar=True, 
#                                      initial_posterior=(post_mean, post_cov), **options)
# mean_cost_history_cov_init = training_history["mean_cost"]
# cost_history_v_cov_init = training_history["voxel_cost"]
# param_history_v_cov_init = training_history["voxel_params"]
# modelfit_cov_init = svb.evaluate(svb.modelfit)
# means_cov_init = svb.evaluate(svb.model_means)
# variances_cov_init = svb.evaluate(svb.model_vars)

if "--show" in sys.argv:
    # Plot some results
    clean_data = data.reshape((-1, num_times))

    for idx in range(num_examples):
        # plt.figure(1)
        # plt.suptitle("Cost history (No covariance)")
        # ax1 = plt.subplot(sq_len, sq_len, idx+1)
        # plt.plot(cost_history_v[idx])

        plt.figure(2)
        plt.suptitle("Cost history (With covariance)")
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(cost_history_v_cov[idx])

        # plt.figure(3)
        # plt.suptitle("Cost history (With covariance, pre-initialized)")
        # ax1 = plt.subplot(sq_len, sq_len, idx+1)
        # plt.plot(cost_history_v_cov_init[idx])

        # plt.figure(4)
        # plt.suptitle("Model fit (No covariance)")
        # ax1 = plt.subplot(sq_len, sq_len, idx+1)
        # plt.plot(svb.data_model.data_flattened[idx],'rx')
        # plt.plot(clean_data[idx], 'g')
        # plt.plot(modelfit[idx],'b')

        plt.figure(5)
        plt.suptitle("Model fit (With covariance)")
        ax1 = plt.subplot(sq_len, sq_len, idx+1)
        plt.plot(svb.data_model.data_flattened[idx],'rx')
        plt.plot(clean_data[idx], 'g')
        plt.plot(modelfit_cov[idx],'b')

        # plt.figure(6)
        # plt.suptitle("Model fit (With covariance, pre-initialized)")
        # ax1 = plt.subplot(sq_len, sq_len, idx+1)
        # plt.plot(svb.data_model.data_flattened[idx],'rx')
        # plt.plot(clean_data[idx], 'g')
        # plt.plot(modelfit_cov_init[idx],'b')

    plt.show()
