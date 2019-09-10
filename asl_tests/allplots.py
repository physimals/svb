"""
Generic figure plotting script
"""

import os
import sys
import math

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
BATCH_SIZES = (5, 6, 9, 12, 18, 24, 48)
SAMPLE_SIZES = (2, 5, 10, 20, 50, 100, 200)
RPTS = (1, 8)
NUM = ("num", "analytic")
COV = ("cov", "nocov")

#BASEDIR = "/mnt/hgfs/win/data/svb/asl"
#MASK = "/mnt/hgfs/win/data/asl/fsl_course/mpld_asltc_mask.nii.gz"
BASEDIR = "c:/Users/ctsu0221/dev/data/svb/asl"
MASK = "c:/Users/ctsu0221/dev/data/asl/fsl_course/mpld_asltc_mask.nii.gz"
MASKDATA = nib.load(MASK).get_data()

DEFAULTS = {
    "lr" : 0.1,
    "bs" : None,
    "ss" : 20,
    "rpts" : 1,
    "num" : "analytic",
    "cov" : "cov",
}

def subdir(params):
    """
    :return: Output subdirectory corresponding to set of parameters
    """
    fullparams = dict(DEFAULTS)
    fullparams.update(params)
    if fullparams["bs"] is None:
        # If batch size unspecified assume no mini-batch optimization
        fullparams["bs"] = fullparams["rpts"]*6
    subdir = os.path.join(BASEDIR, "rpts_%(rpts)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s" % fullparams)
    if not os.path.exists(os.path.join(subdir, "runtime")):
        return None
    else:
        return subdir

def subplot_info(subplots):
    """
    :return nx, ny, sequence of parameter dictionaries for subplots
    """
    if subplots is None:
        nx, ny, items = 1, 1, [{}]
    else:
        if "items" in subplots:
            items = subplots["items"]
        elif "param" in subplots:
            param, values = subplots["param"]
            items = [{param : v} for v in values]

        if "layout" in subplots:
            nx, ny = subplots["layout"]
        else:
            nx = ny = int(math.sqrt(len(items)))
            while nx*ny < len(items):
                if nx <= ny:
                    nx += 1
                else:
                    ny += 1
    return nx, ny, items

def boxplot(subplots, boxes, param, **kwargs):
    """
    Generate a box plot for values of a parameter
    """
    nx, ny, items = subplot_info(subplots)
    plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

        # Full repeat data only done on sample sizes up to 20
        if params["rpts"] == 8 and "ss" in params:
            params["ss"] = min(params["ss"], 20) 
            
        box_param = boxes[0]
        data, labels = [], []
        for box_val in boxes[1]:
            params = dict(params)
            params.update(kwargs)
            params[box_param] = box_val

            if isinstance(box_val, float):
                label = "%.3f" % box_val
            else:
                label = str(box_val)

            if subdir(params) is None:
                continue

            data.append(nib.load(os.path.join(subdir(params), "mean_%s.nii.gz" % param)).get_data()[MASKDATA != 0])
            labels.append(label)

        plt.boxplot(data, labels=labels, showfliers=False)
        title = kwargs.get("title", "") % params
        plt.title(title)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", param))
        plt.xlabel(kwargs.get("xlabel", box_param))
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done box plot: %s" % title)

def line(subplots, lines, **kwargs):
    """
    Generate line plots
    """
    nx, ny, items = subplot_info(subplots)
    plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)
        # Full repeat data only done on sample sizes up to 20
        if params["rpts"] == 8 and "ss" in params:
            params["ss"] = min(params["ss"], 20) 

        line_param = lines[0]
        for line_val in lines[1]:
            params = dict(params)
            params.update(kwargs)
            params[line_param] = line_val

            if isinstance(line_val, float):
                label = "%s: %.3f" % (line_param, line_val)
            else:
                label = "%s: %s" % (line_param, str(line_val))

            if "points4d" in kwargs:
                if subdir(params) is None:
                    continue
                points4d = nib.load(os.path.join(subdir(params), "%s.nii.gz" % kwargs.get("points4d"))).get_data()
                points = np.mean(points4d[MASKDATA != 0], axis=0)
            elif "points_param" in kwargs and "points_values" in kwargs:
                points = []
                for point_val in kwargs["points_values"]:
                    params[kwargs["points_param"]] = point_val
                    if subdir(params) is None:
                        points.append(None)
                        continue
                    points3d = nib.load(os.path.join(subdir(params), "%s.nii.gz" % kwargs.get("points3d"))).get_data()
                    if points3d.ndim == 4:
                        # For 4D data take the last volume
                        points3d = points3d[..., -1]
                    points.append(np.mean(points3d[MASKDATA != 0]))
                plt.xticks(range(len(kwargs["points_values"])), ["%.2f" % v for v in kwargs["points_values"]])
            else:
                raise RuntimeError("No points to plot")

            if kwargs.get("runtime", None):
                with open(os.path.join(subdir(params), "runtime")) as runtime_file:
                    xscale = float(len(points)) / float(runtime_file.read())
                plt.xlabel(kwargs.get("xlabel", "Run time (s)"))
            else:
                xscale = 1.0
                plt.xlabel(kwargs.get("xlabel", ""))
            plt.plot([float(v) / xscale for v in range(len(points))], points, label=label)

        title = kwargs.get("title", "") % params
        plt.title(title)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", ""))
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if len(lines[1]) > 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done line plot: %s" % title)

# Convergence of cost by learning rate
line(
    title="Convergence of cost (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    savename="conv_lr_cov",
)

line(
    title="Convergence of cost (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="nocov",
    savename="conv_lr_nocov",
)

# Best cost by learning rate for various sample sizes
line(
    title="Best cost by learning rate (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    xlabel="Learning rate",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_ss_cov",
)

line(
    title="Best cost by learning rate (no covariance): RPTS=%(rpts)i",
    xlabel="Cost",
    ylabel="Learning rate",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_ss_nocov",
)

# Convergence of cost by batch size
line(
    title="Convergence of cost by batch size (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="cov",
    savename="conv_bs_cov",
)

line(
    title="Convergence of cost by batch size (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="nocov",
    savename="conv_bs_nocov",
)

# Best cost by learning rate for various batch sizes
line(
    title="Best cost by batch size (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    xlabel="Learning rate",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_bs_cov",
)

line(
    title="Best cost by batch size (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_bs_nocov",
)

# Convergence of cost by sample size
line(
    title="Convergence by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    cov="cov",
    savename="conv_ss_cov",
)

line(
    title="Convergence by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    cov="nocov",
    savename="conv_ss_nocov",
)

line(
    title="Mini-batch convergence by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_bs_6_cov",
)

line(
    title="Mini-batch convergence by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    xlabel="Sample size",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_bs_6_nocov",
)

# Convergence of parameters by sample size
boxplot(
    title="ftiss by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_amp1_cov",
)

boxplot(
    title="ftiss by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_amp1_nocov",
)

boxplot(
    title="delttiss by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-4, 5),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="delttiss",
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_amp2_cov",
)

boxplot(
    title="delttiss by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-4, 5),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="delttiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_amp2_nocov",
)

"""
boxplot(
    title="noise by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    #ylim=(-1, 2),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_r1_cov",
)

boxplot(
    title="noise by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-1, 2),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_r1_nocov",
)
"""

# Numerical vs analytic approach
line(
    title="Best cost by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_ss_num_cov",
)

line(
    title="Best cost by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Cost",
    #ylim=(0, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_ss_num_nocov",
)

boxplot(
    title="ftiss by sample size (analytic with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="cov",
    num="analytic",
    savename="conv_ss_amp1_analytic_cov",
)

boxplot(
    title="ftiss by sample size (numerical with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="cov",
    num="num",
    savename="conv_ss_amp1_num_cov",
)

boxplot(
    title="ftiss by sample size (analytic no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    num="analytic",
    savename="conv_ss_amp1_analytic_nocov",
)

boxplot(
    title="ftiss by sample size (numerical no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    num="num",
    savename="conv_ss_amp1_num_nocov",
)

"""
boxplot(
    title="noise by sample size (analytic with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="cov",
    num="analytic",
    savename="conv_ss_r1_analytic_cov",
)

boxplot(
    title="noise by sample size (numerical with covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="cov",
    num="num",
    savename="conv_ss_r1_num_cov",
)
boxplot(
    title="noise by sample size (analytic no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="nocov",
    num="analytic",
    savename="conv_ss_r1_analytic_nocov",
)
boxplot(
    title="noise by sample size (numerical no covariance): RPTS=%(rpts)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="noise",
    lr=0.1,
    bs=6,
    cov="nocov",
    num="num",
    savename="conv_ss_r1_num_nocov",
)
"""
if "--show" in sys.argv:
    plt.show()