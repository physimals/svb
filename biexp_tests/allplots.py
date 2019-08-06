"""
Generic figure plotting script
"""

import os
import sys
import math

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

LEARNING_RATES = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
BATCH_SIZES = (5, 10, 20, 50, 100)
SAMPLE_SIZES = (2, 5, 10, 20, 50, 100, 200)
NT = (10, 20, 50, 100)
NUM = ("num", "analytic")
COV = ("cov", "nocov")
BASEDIR = "/mnt/hgfs/win/data/svb"

DEFAULTS = {
    "lr" : 0.1,
    "bs" : None,
    "ss" : 200,
    "nt" : 100,
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
        fullparams["bs"] = fullparams["nt"]
    return os.path.join(BASEDIR, "nt_%(nt)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s" % fullparams)

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
                nx += 1
                ny += 1
    return nx, ny, items

def boxplot(subplots, boxes, param, **kwargs):
    """
    Generate a box plot for values of a parameter
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, items = subplot_info(subplots)

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

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

            data.append(nib.load(os.path.join(subdir(params), "mean_%s.nii.gz" % param)).get_data().flatten())
            labels.append(label)

        plt.boxplot(data, labels=labels, showfliers=False)
        plt.title(kwargs.get("title", "") % params)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", param))
        plt.xlabel(kwargs.get("xlabel", box_param))
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
            if isinstance(ylim, list):
                ylim = ylim[idx]
            plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))

def line(subplots, lines, **kwargs):
    """
    Generate line plots
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, items = subplot_info(subplots)

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

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
                if not os.path.exists(subdir(params)):
                    continue
                points4d = nib.load(os.path.join(subdir(params), "%s.nii.gz" % kwargs.get("points4d"))).get_data()
                points = np.mean(points4d, axis=(0, 1, 2))
            elif "points_param" in kwargs and "points_values" in kwargs:
                points = []
                for point_val in kwargs["points_values"]:
                    params[kwargs["points_param"]] = point_val
                    if not os.path.exists(subdir(params)):
                        points.append(None)
                        continue
                    points3d = nib.load(os.path.join(subdir(params), "%s.nii.gz" % kwargs.get("points3d"))).get_data()
                    if points3d.ndim == 4:
                        # For 4D data take the last volume
                        points3d = points3d[..., -1]
                    points.append(np.mean(points3d))
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

        plt.title(kwargs.get("title", "") % params)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", ""))
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
            if isinstance(ylim, list):
                ylim = ylim[idx]
            plt.ylim(ylim)
        if len(lines[1]) > 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))

# Convergence of cost by learning rate
line(
    title="Convergence of cost (with covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    savename="conv_lr_cov",
)

line(
    title="Convergence of cost (no covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="nocov",
    savename="conv_lr_nocov",
)

# Best cost by learning rate for various sample sizes
line(
    title="Best cost by learning rate (with covariance): NT=%(nt)i",
    xlabel="Cost",
    ylabel="Learning rate",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_ss_cov",
)

line(
    title="Best cost by learning rate (no covariance): NT=%(nt)i",
    xlabel="Cost",
    ylabel="Learning rate",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_ss_nocov",
)

# Convergence of cost by batch size
line(
    title="Convergence of cost by batch size (with covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="cov",
    savename="conv_bs_cov",
)

line(
    title="Convergence of cost by batch size (no covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="nocov",
    savename="conv_bs_nocov",
)

# Best cost by learning rate for various batch sizes
line(
    title="Best cost by batch size (with covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_bs_cov",
)

line(
    title="Best cost by batch size (no covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_bs_nocov",
)

# Convergence of cost by sample size
line(
    title="Convergence by sample size (with covariance): NT=%(nt)i",
    ylabel="Cost",
    xlabel="Sample size",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    cov="cov",
    savename="conv_ss_cov",
)

line(
    title="Convergence by sample size (no covariance): NT=%(nt)i",
    ylabel="Cost",
    xlabel="Sample size",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    cov="nocov",
    savename="conv_ss_nocov",
)
line(
    title="Mini-batch convergence by sample size (with covariance): NT=%(nt)i",
    ylabel="Cost",
    xlabel="Sample size",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    bs=10,
    cov="cov",
    savename="conv_ss_bs_10_cov",
)

line(
    title="Mini-batch convergence by sample size (no covariance): NT=%(nt)i",
    ylabel="Cost",
    xlabel="Sample size",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_bs_10_nocov",
)

# Convergence of parameters by sample size
boxplot(
    title="amp1 by sample size (with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="cov",
    savename="conv_ss_amp1_cov",
)

boxplot(
    title="amp1 by sample size (no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_amp1_nocov",
)

boxplot(
    title="amp2 by sample size (with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp2",
    lr=0.1,
    bs=10,
    cov="cov",
    savename="conv_ss_amp2_cov",
)

boxplot(
    title="amp2 by sample size (no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp2",
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_amp2_nocov",
)

boxplot(
    title="r1 by sample size (with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 2),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="cov",
    savename="conv_ss_r1_cov",
)

boxplot(
    title="r1 by sample size (no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 2),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_r1_nocov",
)

boxplot(
    title="r2 by sample size (with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r2",
    lr=0.1,
    bs=10,
    cov="cov",
    savename="conv_ss_r2_cov",
)

boxplot(
    title="r2 by sample size (no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r2",
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_r2_nocov",
)

# Numerical vs analytic approach
line(
    title="Best cost by sample size (with covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_ss_num_cov",
)

line(
    title="Best cost by sample size (no covariance): NT=%(nt)i",
    ylabel="Cost",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_ss_num_nocov",
)

boxplot(
    title="amp1 by sample size (analytic with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="cov",
    num="analytic",
    savename="conv_ss_amp1_analytic_cov",
)

boxplot(
    title="amp1 by sample size (numerical with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="cov",
    num="num",
    savename="conv_ss_amp1_num_cov",
)

boxplot(
    title="amp1 by sample size (analytic no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="nocov",
    num="analytic",
    savename="conv_ss_amp1_analytic_nocov",
)

boxplot(
    title="amp1 by sample size (numerical no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="amp1",
    lr=0.1,
    bs=10,
    cov="nocov",
    num="num",
    savename="conv_ss_amp1_num_nocov",
)

boxplot(
    title="r1 by sample size (analytic with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="cov",
    num="analytic",
    savename="conv_ss_r1_analytic_cov",
)

boxplot(
    title="r1 by sample size (numerical with covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="cov",
    num="num",
    savename="conv_ss_r1_num_cov",
)

boxplot(
    title="r1 by sample size (analytic no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="nocov",
    num="analytic",
    savename="conv_ss_r1_analytic_nocov",
)

boxplot(
    title="r1 by sample size (numerical no covariance): NT=%(nt)i",
    ylabel="Sample size",
    ylim=(-1, 3),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r1",
    lr=0.1,
    bs=10,
    cov="nocov",
    num="num",
    savename="conv_ss_r1_num_nocov",
)

if "--show" in sys.argv:
    plt.show()