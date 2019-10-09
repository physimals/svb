"""
Run tests on biexponential model
"""
import os
import sys
import math
import glob

import nibabel as nib
import numpy as np
import tensorflow as tf

from svb import dist
import svb.main

import fabber

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

# Output options
BASEDIR = "/mnt/hgfs/win/data/svb/biexp2"

def test_biexp(fname, outdir=".", **kwargs):
    """
    Fit a 4D Nifti image to a biexponential model

    :param fname: File name of Nifti image
    :param outdir: Output directory relative to BASEDIR
    """
    if os.path.exists("logging.conf"):
        kwargs["log_config"] = "logging.conf"

    kwargs.update({
        "output" : os.path.join(BASEDIR, outdir),
        "save_mean" : True,
        "save_var" : True,
        "save_model_fit" : True,
        "save_noise" : True,
        "save_param_history" : True,
        "save_cost" : True,
        "save_cost_history" : True,
    })
    
    _runtime, mean_cost_history = svb.main.run(data_fname=fname, model_name="biexp", tee=sys.stdout, **kwargs)

    normalize_exps(kwargs["output"])
    return mean_cost_history

def normalize_exps(outdir):
    """
    'Normalize' the output of 4-parameter fitting with non-informative priors

    Basically with a biexponential the two exponentials can 'swap'. This
    function simply normalizes so that amp1, r1 is the exponential with
    the lower decay rate at every voxel
    """
    r1 = nib.load(os.path.join(outdir, "mean_r1.nii.gz")).get_data()
    r2 = nib.load(os.path.join(outdir, "mean_r2.nii.gz")).get_data()
    wrong_way_round = r1 > r2
    for param in ["amp", "r"]:
        for output in ["mean", "std"]:
            for hist in ["", "_history"]:
                fname1 = "%s_%s1%s.nii.gz" % (output, param, hist)
                fname2 = "%s_%s2%s.nii.gz" % (output, param, hist)
                fname1_hist = "%s_%s1_history.nii.gz" % (output, param)
                fname2_hist = "%s_%s2_history.nii.gz" % (output, param)
                if os.path.exists(os.path.join(outdir, fname1)):
                    #print("Found ", fname1)
                    out1 = nib.load(os.path.join(outdir, fname1)).get_data()
                    out2 = nib.load(os.path.join(outdir, fname2)).get_data()
                    out1_save = np.copy(out1)
                    out1[wrong_way_round] = out2[wrong_way_round]
                    out2[wrong_way_round] = out1_save[wrong_way_round]
                    #print("Saving new ", os.path.join(outdir, fname1))
                    nib.Nifti1Image(out1, np.identity(4)).to_filename(os.path.join(outdir, fname1))
                    nib.Nifti1Image(out2, np.identity(4)).to_filename(os.path.join(outdir, fname2))

def redo_normalize():
    for output_dir in glob.glob(os.path.join(BASEDIR, "*")):
        if os.path.isdir(output_dir) and os.path.exists(os.path.join(output_dir, "runtime")):
            normalize_exps(output_dir)

def generate_test_data(num_voxels, num_times, dt, m1, m2, l1, l2, noise):
    """
    Create Nifti files containing instances of biexponential data
    with and without noise
    """
    sq_len = int(math.sqrt(num_voxels))
    shape = (sq_len, sq_len, 1, num_times)
    data = np.zeros(shape)
    for t_idx in range(num_times):
        t = t_idx*dt
        data[..., t_idx] = m1*math.exp(-l1*t) + m2*math.exp(-l2*t)
    nii = nib.Nifti1Image(data, np.identity(4))
    nii.to_filename(os.path.join(BASEDIR, FNAME_TRUTH % num_times))
    data_noisy = data + np.random.normal(0, noise, size=shape)
    nii_noisy = nib.Nifti1Image(data_noisy, np.identity(4))
    nii_noisy.to_filename(os.path.join(BASEDIR, FNAME_NOISY % num_times))

def run_fabber():
    for nt, dt in zip(NT, DT):
        outdir=os.path.join(BASEDIR, "fab_nt_%i" % nt)
        if os.path.exists(os.path.join(outdir, "logfile")):
            print("Skipping %s" % outdir)
            continue
        options = {
            "method" : "vb",
            "noise" : "white",
            "model" : "exp",
            "num-exps" : 2,
            "dt" : dt,
            "data" : os.path.join(BASEDIR, FNAME_NOISY % nt),
            "max-iterations" : 100,
            "save-mean" : True,
            "save-var" : True,
            "save-model-fit" : True,
            "save-noise-mean" : True,
            "overwrite" : True,
        }
        fab = fabber.Fabber()
        run = fab.run(options, fabber.percent_progress(), debug=True)
        run.write_to_dir(outdir)
        normalize_exps(outdir)

        print("Done fabber run: nt=%i" % nt)
                        
def run_combinations(**kwargs):
    """
    Run combinations of all test variables

    (excluding prior/posterior tests)
    """
    learning_rates = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (5, 10, 20, 50, 100)
    sample_sizes = (1, 2, 5, 10, 20, 50, 100, 200)
    
    for infer_covar, cov in zip((True, False), ("cov", "nocov")):
        for num_ll, num in zip((False, True), ("analytic", "num")):
            for nt, dt in zip(NT, DT):
                for bs in batch_sizes:
                    if bs > nt:
                        continue
                    for lr in learning_rates:
                        for ss in sample_sizes:
                            outdir="nt_%i_lr_%.3f_bs_%i_ss_%i_%s_%s" % (nt, lr, bs, ss, num, cov)
                            if os.path.exists(os.path.join(BASEDIR, outdir, "runtime")):
                                print("Skipping %s" % outdir)
                                continue
                            mean_cost_history = test_biexp(os.path.join(BASEDIR, FNAME_NOISY % nt),
                                                           t0=0,
                                                           dt=dt,
                                                           outdir=outdir,
                                                           epochs=500,
                                                           batch_size=bs,
                                                           learning_rate=lr,
                                                           sample_size=ss,
                                                           force_num_latent_loss=num_ll,
                                                           infer_covar=infer_covar,
                                                           **kwargs)
                            print(nt, lr, bs, ss, num, cov, mean_cost_history[-1])

def priors_posteriors(suffix="", **kwargs):
    """
    Run tests on various combinations of prior and posterior
    """
    nt, dt = NT[-1], DT[-1]
    test_data = os.path.join(BASEDIR, FNAME_NOISY % nt)
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
                                       epochs=1000,
                                       batch_size=bs,
                                       learning_rate=lr,
                                       lr_quench=0.95,
                                       param_overrides=param_overrides,
                                       **kwargs)
        print(name, mean_cost_history[-1])

if __name__ == "__main__":
    # MC needs this it would appear!
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    #for nt, dt in zip(NT, DT):
    #    generate_test_data(num_voxels=NV, num_times=nt, dt=dt, m1=M1, m2=M2, l1=L1, l2=L2, noise=NOISE)

    #redo_normalize()
    run_combinations()
    #run_fabber()
    #priors_posteriors("_num", infer_covar=False, force_num_latent_loss=True)
    #priors_posteriors("_analytic", infer_covar=False)
    #priors_posteriors("_num_corr", infer_covar=True, force_num_latent_loss=True)
    #priors_posteriors("_analytic_corr", infer_covar=True)
