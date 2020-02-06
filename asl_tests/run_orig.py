"""
Run ASL tests
"""
import os
import sys

import numpy as np
import tensorflow as tf

import svb.main

# Test data properties
FNAME_RPTS = "mpld_asltc_diff.nii.gz"
FNAME_MEAN = "mpld_asltc_diff_mean.nii.gz"
FNAME_MASK = "mpld_asltc_mask.nii.gz"
PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SLICEDT = 0.0452
TAU = 1.8
RPTS = 8

# Optimization properties
EPOCHS = 500

# Output options
INPUTDIR = "/mnt/hgfs/win/data/asl/fsl_course"
BASEDIR = "/mnt/hgfs/win/data/svb/asl"

def test_asl(fname, outdir=".", **kwargs):
    """
    Fit to a 4D ASL data

    :param fname: File name of Nifti image
    """
    if os.path.exists("logging.conf"):
        kwargs["log_config"] = "logging.conf"

    kwargs["output"] = os.path.join(BASEDIR, outdir)
    _runtime, mean_cost_history = svb.main.run(data_fname=fname, model_name="aslrest", tee=sys.stdout, **kwargs)
    return mean_cost_history

def run_best(**kwargs):
    print("Running best effort")
    for infer_covar, cov in zip((True, False), ("cov", "nocov")):
        for num_ll, num in zip((False, True), ("analytic", "num")):
            fname = FNAME_RPTS
            rpts = 8
            bs = 10
            lr = 0.1
            ss = 10
            lr_quench = 1.0
            lr_min = 0.005
            max_trials = 20
            outdir="best_rpts_%i_lr_%.3f_bs_%i_ss_%i_%s_%s" % (rpts, lr, bs, ss, num, cov)
            mean_cost_history = test_asl(os.path.join(INPUTDIR, fname),
                                            mask_fname=os.path.join(INPUTDIR, FNAME_MASK),
                                            plds=PLDS,
                                            slicedt=SLICEDT,
                                            repeats=rpts,
                                            tau=TAU,
                                            outdir=outdir,
                                            epochs=EPOCHS,
                                            batch_size=bs,
                                            learning_rate=lr,
                                            sample_size=ss,
                                            lr_quench=lr_quench,
                                            force_num_latent_loss=num_ll,
                                            infer_covar=infer_covar,
                                            max_trials=max_trials,
                                            lr_min=lr_min,
                                            **kwargs)
            print(rpts, lr, bs, ss, num, cov, mean_cost_history[-1])

def run_combinations(**kwargs):
    """
    Run combinations of all test variables

    (excluding prior/posterior tests)
    """
    learning_rates = (0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (6, 5, 9, 12, 18, 24, 48)
    #sample_sizes = (2, 5, 10, 20, 50, 100, 200)
    sample_sizes = (2, 5, 10, 20)

    for fname, rpts in zip((FNAME_MEAN, FNAME_RPTS), (1, 8)):
        for ss in sample_sizes:
            for infer_covar, cov in zip((True, False), ("cov", "nocov")):
                for num_ll, num in zip((False, True), ("analytic", "num")):
                    for bs in batch_sizes:
                        if bs > 6*rpts:
                            continue
                        for lr in learning_rates:
                            outdir="rpts_%i_lr_%.3f_bs_%i_ss_%i_%s_%s" % (rpts, lr, bs, ss, num, cov)
                            runtime = os.path.join(BASEDIR, outdir, "runtime")
                            mean_ftiss = os.path.join(BASEDIR, outdir, "mean_ftiss.nii.gz")
                            if os.path.exists(runtime) and os.path.exists(mean_ftiss):
                                mtime = os.path.getmtime(runtime)
                                import calendar
                                import time
                                fromtime = time.strptime('Jul 15, 2019', '%b %d, %Y')
                                if mtime > calendar.timegm(fromtime):
                                    print("After - skipping", outdir, mtime)
                                    continue
                                else:
                                    print("Before", outdir, mtime)
                            mean_cost_history = test_asl(fname,
                                                         mask=FNAME_MASK,
                                                         plds=PLDS,
                                                         slicedt=SLICEDT,
                                                         repeats=rpts,
                                                         tau=TAU,
                                                         outdir=outdir,
                                                         epochs=EPOCHS,
                                                         batch_size=bs,
                                                         learning_rate=lr,
                                                         sample_size=ss,
                                                         lr_quench=1.0,
                                                         force_num_latent_loss=num_ll,
                                                         infer_covar=infer_covar,
                                                         **kwargs)
                            print(rpts, lr, bs, ss, num, cov, mean_cost_history[-1])

def mean_learning_rate():
    """
    Run test on mean across repeats varying learning rate
    """
    learning_rates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)

    for lr in learning_rates:
        mean_cost_history = test_asl(
            fname=FNAME_MEAN,
            mask=FNAME_MASK,
            plds=PLDS,
            slicedt=SLICEDT,
            repeats=1,
            tau=TAU,
            outdir="mean_lr_%.3f" % lr,
            epochs=EPOCHS,
            batch_size=len(PLDS),
            learning_rate=lr,
            lr_quench=0.95)

        print(lr, mean_cost_history[-1])

def rpts_learning_rate():
    """
    Run test on full repeated data varying learning rate and batch size
    """
    learning_rates = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (5, 6, 10, 12, 20, 24)

    for bs in batch_sizes:
        for lr in learning_rates:
            mean_cost_history = test_asl(
                fname=FNAME_RPTS,
                mask=FNAME_MASK,
                plds=PLDS,
                slicedt=SLICEDT,
                repeats=RPTS,
                tau=TAU,
                outdir="rpts_lr_%.3f_bs_%i" % (lr, bs),
                epochs=EPOCHS,
                batch_size=bs,
                learning_rate=lr,
                lr_quench=0.95)

            print(lr, bs, mean_cost_history[-1])

def _run():
    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    #mean_learning_rate()
    #rpts_learning_rate()
    #run_combinations()
    run_best()

if __name__ == "__main__":
    _run()
