import os
import sys
import math
import logging, logging.config

import nibabel as nib
import numpy as np
import tensorflow as tf

sys.path.append("..")
from svb.models.exp_models import BiExpModel
from svb.svb import SvbFit
import svb.dist as dist

# Properties of the ground truth data
M1 = 10
M2 = 10
L1 = 1
L2 = 10

# Properties of the test data
FNAME_TRUTH = "sim_data_biexp_%i_truth.nii.gz"
FNAME_NOISY = "sim_data_biexp_%i_noisy.nii.gz"
NV = 1000
DT = (0.5, 0.25, 0.1, 0.05)
NT = (10, 20, 50, 100)
NOISE = 1.0

def test_biexp(fname, t0, dt, mask=None, outdir=".", **kwargs):
    """
    Fit to a 4D Nifti image

    :param fname: File name of Nifti image
    :param mode: Model to fit to
    :param t0: Timeseries first value
    :param dt: Time step
    """
    nii = nib.load(fname)
    d = nii.get_data()
    shape = list(d.shape)
    d_flat = d.reshape(-1, shape[-1])

    if mask is not None:
        mask_nii = nib.load(mask)
        mask_d = mask_nii.get_data()
        mask_flat = mask_d.flatten()
        d_flat = d_flat[mask_flat > 0]

    param_overrides = kwargs.get("param_overrides", None)
    if param_overrides is None:
        param_overrides = {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e1)},
        }
    model = BiExpModel(param_overrides=param_overrides)

    # Generate timeseries FIXME should be derivable from the model
    t = np.linspace(t0, t0+shape[3]*dt, num=shape[3], endpoint=False)

    # Train with no correlation between parameters
    svb = SvbFit(model, **kwargs)
    ret = svb.train(t, d_flat, kwargs.pop("batch_size", 10), **kwargs)
    mean_cost_history = ret[0]

    # Transpose output so the first index is by parameter
    means = svb.output("model_params")
    variances = np.transpose(np.diagonal(svb.output("post_cov"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    makedirs(outdir, exist_ok=True)
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(means[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = nib.Nifti1Image(variances[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out voxelwise cost history
    cost_history_v = ret[2]
    cost_history_v_nii = nib.Nifti1Image(cost_history_v.reshape(shape[:3] + [cost_history_v.shape[1]]),
                                         None, header=nii.get_header())
    cost_history_v_nii.to_filename(os.path.join(outdir, "cost_history.nii.gz"))

    # Write out voxelwise parameter history
    param_history_v = ret[3]
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(param_history_v[:, :, idx].reshape(list(shape[:3]) + [cost_history_v.shape[1]]),
                                   None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s_history.nii.gz" % param.name))

    nii_mean = nib.Nifti1Image(param_history_v[:, :, model.nparams].reshape(list(shape[:3]) + [cost_history_v.shape[1]]),
                               None, header=nii.get_header())
    nii_mean.to_filename(os.path.join(outdir, "mean_noise_history.nii.gz"))

    # Write out modelfit
    fit = ret[4]
    fit_nii = nib.Nifti1Image(fit.reshape(shape), None, header=nii.get_header())
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

    return mean_cost_history

def runtime(runnable, *args, **kwargs):
    import time
    t0 = time.time()
    ret = runnable(*args, **kwargs)
    t1 = time.time()
    return (t1-t0), ret

def makedirs(d, exist_ok=False):
    try:
        os.makedirs(d)
    except OSError as exc:
        import errno
        if not exist_ok or exc.errno != errno.EEXIST:
            raise

def generate_test_data(num_voxels, num_times, dt, m1, m2, l1, l2, noise):
    sq_len = int(math.sqrt(num_voxels))
    shape = (sq_len, sq_len, 1, num_times)
    data = np.zeros(shape)
    for t_idx in range(num_times):
        t = t_idx*dt
        data[..., t_idx] = m1*math.exp(-l1*t) + m2*math.exp(-l2*t)
    nii = nib.Nifti1Image(data, np.identity(4))
    nii.to_filename(FNAME_TRUTH % num_times)
    data_noisy = data + np.random.normal(0, noise, size=shape)
    nii_noisy = nib.Nifti1Image(data_noisy, np.identity(4))
    nii_noisy.to_filename(FNAME_NOISY % num_times)

def learning_rate():
    learning_rates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (5, 10, 15, 25, 50)

    for nt, dt in zip(NT, DT):
        for bs in batch_sizes:
            if bs > nt: continue
            for lr in learning_rates:
                mean_cost_history = test_biexp(FNAME_NOISY % nt, t0=0, dt=dt,
                                               outdir="nt_%i_lr_%.3f_bs_%i" % (nt, lr, bs),
                                               training_epochs=500,
                                               batch_size=bs,
                                               learning_rate=lr,
                                               quench_rate=0.95)
                print(nt, bs, lr, mean_cost_history[-1])

def priors_posteriors(suffix="", **kwargs):
    nt, dt = NT[-1], DT[-1]
    test_data = FNAME_NOISY % nt
    cases = {
        # Non-informative prior and initial posterior
        "prior_ni_post_ni" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
        },
        # Non-informative prior and initial posterior with data-driven initialisation of mean for amp1 and amp2
        "prior_ni_post_ni_init" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 1e6)},
        },
        # Non-informative prior, informative posterior
        "prior_ni_post_i" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
        },
        # Non-informative prior, informative posterior with data-driven initialisation of mean for amp1 and amp2
        "prior_ni_post_i_init" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
        },
        # Non-informative prior, informative posterior set close to true solution
        "prior_ni_post_i_true" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(10.0, 2.0), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(10.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(10.0, 2.0), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
        },
        # Non-informative prior, informative posterior set far from true solution
        "prior_ni_post_i_wrong" : {
            "amp1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(100.0, 2.0), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(100.0, 2.0), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 1e6), "post" : dist.LogNormal(1.0, 2.0)},
        },
        # Informative prior, non-informative posterior
        "prior_i_post_ni" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
        },
        # Informative prior, non-informative posterior with data-driven initialisation of mean for amp1 and amp2
        "prior_i_post_ni_init" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 1e6)},
        },
        # Informative prior, informative posterior 
        "prior_i_post_i" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0), "initialise" : None},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0), "initialise" : None},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
        },
        # Informative prior, informative posterior with data-driven initialisation of mean for amp1 and amp2
        "prior_i_post_i_init" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
        }, 
        # Informative prior, informative posterior set close to true solution
        "prior_i_post_i_true" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(10.0, 2.0)},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(10.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(10.0, 2.0)},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
        }, 
        # Informative prior, informative posterior set far from true solution
        "prior_i_post_i_wrong" : {
            "amp1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(100.0, 2.0)},
            "r1" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
            "amp2" : {"prior" : dist.LogNormal(10.0, 2.0), "post" : dist.LogNormal(100.0, 2.0)},
            "r2" : {"prior" : dist.LogNormal(1.0, 2.0), "post" : dist.LogNormal(1.0, 2.0)},
        },
    }

    bs = 10
    lr = 0.1
    for name, param_overrides in cases.items():
        name = "%s%s" % (name, suffix)
        print(name)
        mean_cost_history = test_biexp(test_data, t0=0, dt=dt,
                                       outdir=name,
                                       training_epochs=1000,
                                       batch_size=bs,
                                       learning_rate=lr,
                                       quench_rate=0.95,
                                       param_overrides=param_overrides,
                                       **kwargs)
        print(name, mean_cost_history[-1])

if __name__ == "__main__":
    # MC needs this it would appear!
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # Logging configuration
    logging.getLogger().setLevel(logging.WARNING)
    if os.path.exists("logging.conf"):
        logging.config.fileConfig("logging.conf")

    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    #for nt, dt in zip(NT, DT):
    #    generate_test_data(num_voxels=NV, num_times=nt, dt=dt, m1=M1, m2=M2, l1=L1, l2=L2, noise=NOISE)
        
    #learning_rate()
    #priors_posteriors("_num", infer_covar=False, force_num_latent_loss=True)
    #priors_posteriors("_analytic", infer_covar=False)
    priors_posteriors("_num_corr", infer_covar=True, force_num_latent_loss=True)
    priors_posteriors("_analytic_corr", infer_covar=True)
