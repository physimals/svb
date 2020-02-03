"""
Run ASL tests
"""
import os
import sys

import numpy as np
import nibabel as nib

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
   
import fabber

import svb.main

# Test data properties
FNAME_RPTS = "mpld_asltc_diff.nii.gz"
FNAME_MEAN = "mpld_asltc_diff_mean.nii.gz"
FNAME_MASK = "mpld_asltc_mask.nii.gz"
PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
SLICEDT = 0.0452
TAU = 1.8
RPTS = 8
CASL = True
BAT = 1.3
BATSD = 0.5

# Optimization properties
EPOCHS = 500

# Output options
BASEDIR = "/mnt/hgfs/win/data/svb/asl"

def run_svb_test(fname, outdir=".", **kwargs):
    """
    Fit to a 4D ASL data

    :param fname: File name of Nifti image
    """
    options = {
        "data" : os.path.join(BASEDIR, fname),
        "output" : os.path.join(BASEDIR, outdir),
        "mask" : FNAME_MASK,
        "model_name" : "aslrest",
        "log_stream" : sys.stdout,
        "plds" : PLDS,
        "slicedt" : SLICEDT,
        "tau" : TAU,
        "casl" : CASL,
        "att" : BAT,
        "attsd" : BATSD,
        "save_mean" : True,
        "save_var" : True,
        "save_model_fit" : True,
        "save_noise" : True,
        "save_param_history" : True,
        "save_cost" : True,
        "save_cost_history" : True,
        "save_log" : True,
        "save_runtime" : True,
    }
    if os.path.exists("logging.conf"):
        options["log_config"] = "logging.conf"
    
    options.update(kwargs)
    return svb.main.run(**options)

def run_fabber_test(fname, outdir, **kwargs):
    options = {
        "data" : os.path.join(BASEDIR, fname),
        "mask" : FNAME_MASK,
        "method" : "vb",
        "noise" : "white",
        "model" : "aslrest",
        "infertiss" : True,
        "inctiss" : True,
        "incbat" : True,
        "inferbat" : True,
        "tau" : TAU,
        "slicedt" : SLICEDT,
        "casl" : CASL,
        "bat" : BAT,
        "batsd" : BATSD,
        "max-iterations" : 100,
        "save-mean" : True,
        "save-std" : True,
        "save-model-fit" : True,
        "save-noise-mean" : True,
        "save-noise-std" : True,
        "overwrite" : True,
    }
    for idx, pld in enumerate(PLDS):
        options["ti%i" % (idx+1)] = pld + TAU

    options.update(kwargs)

    fab = fabber.Fabber()
    run = fab.run(options, fabber.percent_progress(), debug=True)
    run.write_to_dir(outdir, ref_nii=nib.load(os.path.join(BASEDIR, fname)))
    return run

def run_spatial_tests():
    SS = 5
    cases = {
        "SpatialMRF" : {
            "ftiss" : {
                "prior_type" : "M",
            }
        },
        "SpatialMRF2" : {
            "ftiss" : {
                "prior_type" : "M2",
            }
        },
        "SpatialFabberMRF" : {
            "ftiss" : {
                "prior_type" : "Mfab",
            }
        },
        "NonSpatial" : {
        },
    }
    for outdir, param_overrides in cases.items():
        _runtime, _svb, training_history = run_svb_test(
            FNAME_RPTS,
            repeats=[8],
            outdir=outdir,
            param_overrides=param_overrides,
            epochs=500,
            sample_size=SS,
            batch_size=10,
            learrning_rate=0.05,
        )
        print("%s: %f, %f" % (outdir, training_history["mean_cost"][-1], training_history["ak"][-1]))
        with open(os.path.join(BASEDIR, outdir, "ak.txt"), "w") as f:
            for ak in training_history["ak"]:
                f.write("%f\n" % ak)

def run_art_tests():
    SS = 5
    cases = {
        "SpatialMRFArt" : {
            "ftiss" : {
                "prior_type" : "M",
            }
        },
        "NonSpatialArt" : {
        },
    }
    for outdir, param_overrides in cases.items():
        _runtime, _svb, training_history = run_svb_test(
            FNAME_RPTS,
            repeats=[8],
            attsd=0.5,
            inferart=True,
            outdir=outdir,
            param_overrides=param_overrides,
            epochs=500,
            sample_size=SS,
            batch_size=10,
            learrning_rate=0.1,
            revert_post_trials=10,
        )
        print("%s: %f, %f" % (outdir, training_history["mean_cost"][-1], training_history["ak"][-1]))
        with open(os.path.join(BASEDIR, outdir, "ak.txt"), "w") as f:
            for ak in training_history["ak"]:
                f.write("%f\n" % ak)

def plot_spatial_ak():
    from matplotlib import pyplot as plt
    plt.figure(1)
    for outdir in ("SpatialMRF", "SpatialMRF2", "SpatialFabberMRF", "fab_rpts_8_spatial"):
        if outdir.startswith("fab"):
            ak = []
            with open(os.path.join(BASEDIR, outdir, "logfile")) as f:
                for line in f.readlines():
                    idx = line.find("New aK:")
                    if idx >= 0:
                        ak.append(float(line[idx+7:]))
        else:
            with open(os.path.join(BASEDIR, outdir, "ak.txt")) as f:
                ak = [float(ak) for ak in f.readlines()]
        plt.plot(ak, label=outdir)
    plt.legend()
    plt.show()

def run_decay_rate_tests(**kwargs):
    """
    Run tests of decay_rateed vs non-decay_rateed optimization
    """
    learning_rates = (0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    decay_rates = (1.0, 0.1, 0.5)
    fname, rpts = FNAME_RPTS, 8
    infer_covar = True
    num_ll = False
    bs = 9
    ss = 10

    for lr in learning_rates:
        for lr_decay_rate in decay_rates:
            outdir="qtests4_lr_%.3f_bs_%i_ss_%i_dr_%.3f" % (lr, bs, ss, lr_decay_rate)
            
            if os.path.exists(os.path.join(BASEDIR, outdir, "runtime")):
                print("Skipping %s" % outdir)
                continue
            _runtime, _svb, training_history = run_svb_test(
                fname,
                repeats=rpts,
                outdir=outdir,
                epochs=EPOCHS,
                batch_size=bs,
                learning_rate=lr,
                sample_size=ss,
                lr_decay_rate=lr_decay_rate,
                force_num_latent_loss=num_ll,
                infer_covar=infer_covar,
                **kwargs
            )
            print(rpts, lr, bs, ss, training_history["mean_cost"][-1])

def run_sample_size_increase_tests(**kwargs):
    """
    Run tests of decay_rateed vs non-decay_rateed optimization
    """
    learning_rate = 0.1
    fname, rpts = FNAME_RPTS, 8
    infer_covar = True
    num_ll = False
    bs = 9
    sample_sizes = (2, 5, 10, 20, 50, 100)
    increase_factors = (1.0, 1.5, 2.0, 5.0)
    
    for ss in sample_sizes:
        for ss_increase_factor in increase_factors:
            outdir="sstests_lr_%.3f_bs_%i_ss_%i_if_%.3f" % (lr, bs, ss, increase_factors)
            
            if os.path.exists(os.path.join(BASEDIR, outdir, "runtime")):
                print("Skipping %s" % outdir)
                continue
            _runtime, _svb, training_history = run_svb_test(
                fname,
                repeats=[rpts],
                outdir=outdir,
                epochs=EPOCHS,
                batch_size=bs,
                learning_rate=lr,
                sample_size=ss,
                ss_increase_factor=ss_increase_factor,
                force_num_latent_loss=num_ll,
                infer_covar=infer_covar,
                **kwargs
            )
            print(rpts, lr, bs, ss, training_history["mean_cost"][-1])

def run_fabber(**kwargs):
    cases = {
        "spatial" : {
            "method" : "spatialvb",
            "param-spatial-priors" : "MN+",
        },
        "nonspatial" : {
        },
        "spatial_art" : {
            "method" : "spatialvb",
            "param-spatial-priors" : "MN+",
            "inferart" : True,
        },
        "nonspatial_art" : {
            "inferart" : True,
        },
    }

    for outname, options in cases.items():
        for fname, rpts in zip((FNAME_MEAN, FNAME_RPTS), (1, 8)):
            outdir=os.path.join(BASEDIR, "fab_rpts_%i_%s" % (rpts, outname))
            if os.path.exists(os.path.join(outdir, "logfile")):
                print("Skipping %s" % outdir)
                continue
            run_fabber_test(
                fname, 
                outdir,
                repeats=rpts,
                **options  
            )
            print("Done fabber run: rpts=%i" % rpts)

def run_combinations(**kwargs):
    """
    Run combinations of all test variables

    (excluding prior/posterior tests)
    """
    learning_rates = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
    batch_sizes = (48, 24, 18, 12, 9, 6, 5)
    #sample_sizes = (2, 5, 10, 20, 50, 100, 200)
    sample_sizes = (2, 5, 10, 20)

    for num_ll, num in zip((False, True), ("analytic", "num")):
        for fname, rpts in zip((FNAME_MEAN, FNAME_RPTS), (1, 8)):
            for bs in batch_sizes:
                if bs > 6*rpts:
                    continue
                for ss in sample_sizes:
                    for infer_covar, cov in zip((True, False), ("cov", "nocov")):
                        for lr in learning_rates:
                            outdir="rpts_%i_lr_%.3f_bs_%i_ss_%i_%s_%s" % (rpts, lr, bs, ss, num, cov)
                            if os.path.exists(os.path.join(BASEDIR, outdir, "mean_noise.nii.gz")):
                                print("Skipping %s" % outdir)
                                continue
                            _runtime, _svb, training_history = run_svb_test(
                                fname,
                                repeats=[rpts],
                                outdir=outdir,
                                epochs=EPOCHS,
                                batch_size=bs,
                                learning_rate=lr,
                                sample_size=ss,
                                force_num_latent_loss=num_ll,
                                infer_covar=infer_covar,
                                **kwargs
                            )
                            print(rpts, lr, bs, ss, num, cov, training_history["mean_cost"][-1])

def _run():
    # MC needs this it would appear!
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    if "--fabber" in sys.argv:
        run_fabber()
    if "--combinations" in sys.argv:
        run_combinations()
    if "--decay-rate" in sys.argv:
        run_decay_rate_tests()
    if "--sample-size" in sys.argv:
        run_sample_size_increase_tests()
    if "--spatial" in sys.argv:
        run_spatial_tests()
    if "--art" in sys.argv:
        run_art_tests()
    if "--plot-spatial" in sys.argv:
        plot_spatial_ak()

if __name__ == "__main__":
    _run()
