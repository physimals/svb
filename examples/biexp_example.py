import sys
import logging

import numpy as np
import matplotlib.pyplot as plt

from svb import SvbFit, dist
from svb.models.exp import BiExpModel

log = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(levelname)s : %(message)s'))
root = logging.getLogger()
root.setLevel(logging.INFO)
root.addHandler(handler)

# set the properties of the simulated data
n_samples=100
true_params = {
    "amp1" : 10.0, # the true amplitude of the model
    "amp2" : 10.0,
    "r1" : 1.0, # the true decay rate
    "r2" : 10.0,
    "noise_sd" : 1.0,
}
true_var = true_params["noise_sd"]**2

# For time being, make sure we have a good range of samples based on the choice of R1
dt = 5.0/(true_params["r1"] * n_samples)

model = BiExpModel(dt=dt)
log.info("Created model: %s", str(model))

tpts = model.tpts(n_tpts=n_samples, shape=[1])
clean, test_data = model.test_data(tpts, true_params)

# Train model
kwargs = {
    "learning_rate" : 0.02,
    "batch_size" : 10,
    "sample_size" : 10,
    "epochs" : 1000,
}
kwargs["param_overrides"] = {
    "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1), "initialise" : None},
    "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1)},
    "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1), "initialise" : None},
    "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1)},
}

svb = SvbFit(model, **kwargs)
log.info("Training model...")
ret = svb.train(tpts, test_data, **kwargs)

# Get output, transposing as required so first index is by parameter
mean_cost_history = ret[0]
cost_history_v = ret[2]
param_history_v = ret[3]
modelfit = ret[4]
means = svb.output("model_params")
variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

svb = SvbFit(model, infer_covar=True, **kwargs)
log.info("Training model (with covariance)...")
ret = svb.train(tpts, test_data, **kwargs)

# Get output, transposing as required so first index is by parameter
mean_cost_history_cov = ret[0]
cost_history_v_cov = ret[2]
param_history_v_cov = ret[3]
modelfit_cov = ret[4]
means_cov = svb.output("model_params")
variances_cov = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))


#%% plot cost history
plt.figure(3)

ax1 = plt.subplot(1,2,1)
plt.cla()

plt.subplots_adjust(hspace=2)
plt.subplots_adjust(wspace=0.8)

plt.plot(mean_cost_history)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title('No post correlation')

ax2 = plt.subplot(1,2,2)
plt.cla()

plt.plot(mean_cost_history_cov)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title('Infer post correlation')

#%% plot the estimated functions overlaid on data

plt.figure(4)

ax1 = plt.subplot(1,2,1)
plt.cla()

# plot the data
plt.plot(tpts,test_data[0],'rx')
# plot the ground truth
plt.plot(tpts,clean[0],'r')
plt.plot(tpts,modelfit[0],'b')

plt.figure(4)

ax1 = plt.subplot(1,2,2)
plt.cla()

# plot the data
plt.plot(tpts,test_data[0],'rx')
# plot the ground truth
plt.plot(tpts,clean[0],'r')
plt.plot(tpts,modelfit_cov[0],'b')

plt.show()
