"""
Generic figure plotting script
"""

import os
import sys
import math

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib

LEARNING_RATES = (0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005)
BATCH_SIZES = (5, 10, 20, 50, 100)
SAMPLE_SIZES = (1, 2, 5, 10, 20, 50, 100, 200)
NOISE = (1.0, 2.0, 5.0, 10.0)
NT = (10, 20, 50, 100)
NUM = ("num", "analytic")
COV = ("cov", "nocov")
BASEDIR = "c:/Users/ctsu0221/dev/data/svb/biexp"
#BASEDIR = "/mnt/hgfs/win/data/svb/biexp"

DEFAULTS = {
    "lr" : 0.1,
    "bs" : None,
    "ss" : 200,
    "nt" : 100,
    "num" : "analytic",
    "cov" : "cov",
    "noise" : 1.0,
}

LABELS = {
    "ss" : r"Sample size $L$",
    "bs" : r"Batch size $B$",
    "lr" : r"$\alpha$",
    "num" : r"$N$",
    "amp1" : r"$A_1$",
    "amp2" : r"$A_2$",
    "r1" : r"$R_1$",
    "r2" : r"$R_2$",
    "nt" : r"$N$",
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
    subdir_format = params.get("subdir_format", "nt_%(nt)i_noise_%(noise).1f_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s")
    #subdir_format = params.get("subdir_format", "nt_%(nt)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s")
    subdir = os.path.join(BASEDIR, subdir_format % fullparams)
    if not os.path.exists(os.path.join(subdir, "logfile")):
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

def boxplot(subplots, boxes, **kwargs):
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
            param = params["param"]

            if isinstance(box_val, float):
                label = "%.3f" % box_val
            else:
                label = str(box_val)

            if subdir(params) is None:
                continue

            labels.append(label)
            data_file = os.path.join(subdir(params), "mean_%s.nii.gz" % param)
            if os.path.exists(data_file):
                param_data = nib.load(data_file).get_data().flatten()
                if param == "noise" and "fab" not in subdir(params):
                    param_data = np.exp(param_data)
                data.append(param_data)
            else:
                print("WARNING: not found", data_file)
                data.append([])

        if len(data) == 0:
            continue

        plt.boxplot(data, labels=labels, showfliers=kwargs.get("showfliers", False))
        title = kwargs.get("title", "") % params
        plt.title(title)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", param))
        plt.xlabel(kwargs.get("xlabel", LABELS[box_param]))
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
            if isinstance(ylim, list):
                ylim = ylim[idx]
            plt.ylim(ylim)

    if kwargs.get("tight", True):
        plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done box plot: %s" % title)

def line(subplots, lines, **kwargs):
    """
    Generate line plots
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, items = subplot_info(subplots)

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

        if lines is None:
            lines = ("", [1])
        line_param = lines[0]
        for line_val in lines[1]:
            params = dict(params)
            params.update(kwargs)
            label = ""

            if line_param:
                params[line_param] = line_val
                if isinstance(line_val, float):
                    label = "%s=%.3f" % (LABELS[line_param], line_val)
                else:
                    label = "%s=%s" % (LABELS[line_param], str(line_val))

            if "points4d" in kwargs:
                if subdir(params) is None:
                    continue
                points4d = nib.load(os.path.join(subdir(params), "%s.nii.gz" % kwargs.get("points4d"))).get_data()
                points = np.mean(points4d, axis=(0, 1, 2))
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

            if kwargs.get("normalize_mean", False) and len(points) > 0:
                mean = sum([p for p in points if p is not None]) / len(points)
                new_points = []
                for point in points:
                    if point is None:
                        new_points.append(None)
                    else:
                        new_points.append(point - mean)
                points = new_points

            plt.plot([float(v) / xscale for v in range(len(points))], points, label=label)

        title = kwargs.get("title", "") % params
        plt.title(title)
        #plt.yscale("symlog")
        plt.ylabel(kwargs.get("ylabel", ""))
        if "ylim" in kwargs:
            try:
                # Single set of ylimits for each subplot
                y0 = float(kwargs["ylim"][0])
                plt.ylim(kwargs["ylim"])
            except:
                # Variable ylimits for each subplot
                plt.ylim(kwargs["ylim"][idx])
                
        if len(lines[1]) > 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done line plot: %s" % title)

def conv_speed(subplots, **kwargs):
    nx, ny, items = subplot_info(subplots)
    plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)
        params = dict(params)
        params.update(kwargs)
    
        very_best_cost = 1e99
        mean_cost_histories = []
        for point_val in kwargs["points_values"]:
            print(subdir(params))
            params[kwargs["points_param"]] = point_val
            if subdir(params) is None:
                mean_cost_histories.append(None)
                continue

            cost_history = nib.load(os.path.join(subdir(params), "cost_history.nii.gz")).get_data()
            mean_cost_history = np.mean(cost_history, axis=(0, 1, 2))       
            mean_cost_histories.append(mean_cost_history)

            best_cost = mean_cost_history[-1]
            if best_cost < very_best_cost:
                very_best_cost = best_cost
            
        #print("Very best free energy=", very_best_cost)
        for percentage_factor in (1.02, 1.05, 1.1, 1.2):
            label = "Within %i%% of best free energy" % int((percentage_factor - 1)*100+0.4)
            points = []
           
            for idx, point_val in enumerate(kwargs["points_values"]):
                params[kwargs["points_param"]] = point_val
                if subdir(params) is None:
                    points.append(None)
                    continue

                mean_cost_history = mean_cost_histories[idx]
                best_cost = mean_cost_history[-1]
                #converged_cost = best_cost * percentage_factor
                converged_cost = very_best_cost * percentage_factor
                if mean_cost_history[-1] > converged_cost:
                    # Did not converge to within this limit
                    points.append(None)
                else:
                    converged = mean_cost_history < converged_cost
                    epoch_converged = np.argmax(converged)
                    #print("Converged at epoch: ", epoch_converged, " of ", len(mean_cost_history))
                    
                    with open(os.path.join(subdir(params), "runtime")) as runtime_file:
                        time_converged = epoch_converged * float(runtime_file.read()) / len(mean_cost_history)
                    
                    #print("Converged at time: ", time_converged)
                    #slow_converged = np.count_nonzero(epoch_converged > 200)
                    points.append(time_converged)
                
            plt.plot(kwargs["points_values"], points, label=label)

        #plt.xticks(range(len(kwargs["points_values"])), ["%.2f" % v for v in kwargs["points_values"]])
        title = kwargs.get("title", "") % params
        plt.title(title)
        plt.ylabel(kwargs.get("ylabel", ""))
        plt.xscale('log')
        plt.xticks(kwargs["points_values"])
        plt.xlabel(kwargs.get("xlabel", ""))
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        if "ylim" in kwargs:
            try:
                # Single set of ylimits for each subplot
                y0 = float(kwargs["ylim"][0])
                plt.ylim(kwargs["ylim"])
            except:
                # Variable ylimits for each subplot
                plt.ylim(kwargs["ylim"][idx])

        plt.legend()

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done conv speed plot: %s" % title)

def param_conv(subplots, samples, param, **kwargs):
    """
    Generate a box plot for values of a parameter
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, items = subplot_info(subplots)

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

        params = dict(params)
        params.update(kwargs)

        if subdir(params) is None:
            continue
        data = nib.load(os.path.join(subdir(params), "mean_%s_history.nii.gz" % param)).get_data()
        nepochs = data.shape[-1]
        data = data.reshape(-1, nepochs)

        #with open(os.path.join(subdir(params), "runtime")) as runtime_file:
        #   runtime = float(runtime_file.read())

        box_data, labels = [], []
        for sample_idx in range(samples):
            epoch = int((nepochs-1) * sample_idx / (samples - 1))
            
            labels.append("%i" % epoch)
            box_data.append(data[:, epoch])
            
        plt.boxplot(box_data, labels=labels, showfliers=False, showmeans=True)
        title = kwargs.get("title", "") % params
        plt.title(title)
        plt.ylabel(kwargs.get("ylabel", param))
        plt.xlabel(kwargs.get("xlabel", "Taining epochs"))
        if "ylim" in kwargs:
            ylim = kwargs["ylim"]
            if isinstance(ylim, list):
                ylim = ylim[idx]
            plt.ylim(ylim)

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done param conv plot: %s" % title)

# Best free energy by learning rate for various sample sizes
line(
    title="Best free energy by learning rate (with covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
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
    title="Best free energy by learning rate (no covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_ss_nocov",
)

# Best free energy by learning rate for various sample sizes at different SNRs

line(
    title="Best free energy by learning rate (with covariance): NT=10, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(10, 40), (20, 50), (25, 55), (30, 60)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=10,
    cov="cov",
    savename="best_cost_lr_ss_nt_10_snr_cov",
)

line(
    title="Best free energy by learning rate (no covariance): NT=10, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(10, 40), (15, 45), (20, 50), (30, 60)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=10,
    cov="nocov",
    savename="best_cost_lr_ss_nt_10_snr_nocov",
)

line(
    title="Best free energy by learning rate (with covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(20, 50), (30, 60), (45, 75), (60, 90)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=20,
    cov="cov",
    savename="best_cost_lr_ss_nt_20_snr_cov",
)

line(
    title="Best free energy by learning rate (no covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(20, 50), (30, 60), (45, 75), (60, 90)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=20,
    cov="nocov",
    savename="best_cost_lr_ss_nt_20_snr_nocov",
)

line(
    title="Best free energy by learning rate (with covariance): NT=50, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(30, 100), (60, 130), (110, 180), (140, 210)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=50,
    cov="cov",
    savename="best_cost_lr_ss_nt_50_snr_cov",
)

line(
    title="Best free energy by learning rate (no covariance): NT=50, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(40, 90), (70, 120), (110, 160), (140, 190)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=50,
    cov="nocov",
    savename="best_cost_lr_ss_nt_50_snr_nocov",
)

line(
    title="Best free energy by learning rate (with covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(50, 110), (130, 190), (210, 270), (280, 340)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=100,
    cov="cov",
    savename="best_cost_lr_ss_nt_100_snr_cov",
)

line(
    title="Best free energy by learning rate (no covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    ylim=[(50, 110), (130, 190), (210, 270), (280, 340)],
    subplots={"param" : ("noise", NOISE),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    nt=100,
    cov="nocov",
    savename="best_cost_lr_ss_nt_100_snr_nocov",
)


# Convergence speed by sample size
conv_speed(
    title="Convergence by sample size (with covariance): N=%(nt)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("nt", NT),},
    points_param="ss",
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="cov",
    savename="conv_speed_ss_cov",
)

conv_speed(
    title="Convergence by sample size (no covariance): N=%(nt)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("nt", NT),},
    points_param="ss",
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="nocov",
    savename="conv_speed_ss_nocov",
)

# Convergence speed by sample size and SNR
conv_speed(
    title="Convergence by sample size (with covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("noise", NOISE),},
    points_param="ss",
    nt=50,
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="cov",
    savename="conv_speed_ss_nt_50_snr_cov",
)

conv_speed(
    title="Convergence by sample size (no covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("noise", NOISE),},
    points_param="ss",
    nt=50,
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="nocov",
    savename="conv_speed_ss_nt_50_snr_nocov",
)

conv_speed(
    title="Convergence by sample size (with covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("noise", NOISE),},
    points_param="ss",
    nt=10,
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="cov",
    savename="conv_speed_ss_nt_10_snr_cov",
)

conv_speed(
    title="Convergence by sample size (no covariance): NT=%(nt)i, noise=%(noise).1f",
    figsize=(13, 10),
    ylabel=r"Time to convergence $s$",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("noise", NOISE),},
    points_param="ss",
    nt=10,
    points_values=SAMPLE_SIZES,
    lr=0.05,
    cov="nocov",
    savename="conv_speed_ss_nt_10_snr_nocov",
)


# Convergence of free energy by sample size
"""line(
    title="Convergence by sample size (with covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel="Runtime (s)",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="cov",
    savename="conv_ss_cov",
)

line(
    title="Convergence by sample size (no covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel="Runtime (s)",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("ss", SAMPLE_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="nocov",
    savename="conv_ss_nocov",
)
"""

# Convergence of free energy by learning rate
line(
    title="Convergence of free energy (with covariance): N=%(nt)i",
    ylabel="Free energy",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("lr", LEARNING_RATES[2:]),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    ss=10,
    savename="conv_lr_cov",
)

line(
    title="Convergence of free energy (no covariance): N=%(nt)i",
    ylabel="Free energy",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("lr", LEARNING_RATES[1:]),
    points4d="cost_history",
    runtime=True,
    cov="nocov",
    ss=10,
    savename="conv_lr_nocov",
)

plt.show()
sys.exit(0)
"""
# Convergence of free energy by batch size
line(
    title="Convergence of free energy by batch size (with covariance): N=%(nt)i",
    ylabel="Free energy",
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
    title="Convergence of free energy by batch size (no covariance): N=%(nt)i",
    ylabel="Free energy",
    ylim=[(10, 60), (20, 90), (30, 150), (60, 250)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="nocov",
    savename="conv_bs_nocov",
)
"""

# Best free energy by learning rate for various batch sizes
line(
    title="Best free energy by batch size (with covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Batch size $B$",
    ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    ss=20,
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_bs_cov",
)

line(
    title="Best free energy by batch size (no covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Batch size $B$",
    ylim=[(15, 25), (22, 40), (39, 90), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("bs", BATCH_SIZES),
    ss=20,
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_bs_nocov",
)

# Convergence speed by batch size
conv_speed(
    title="Convergence by batch size (with covariance): N=%(nt)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Batch size $B$",
    subplots={"param" : ("nt", NT),},
    points_param="bs",
    ss=20,
    points_values=BATCH_SIZES,
    lr=0.05,
    cov="cov",
    savename="conv_speed_bs_cov",
)

conv_speed(
    title="Convergence by batch size (no covariance): N=%(nt)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Batch size $B$",
    subplots={"param" : ("nt", NT),},
    points_param="bs",
    ss=20,
    points_values=BATCH_SIZES,
    lr=0.05,
    cov="nocov",
    savename="conv_speed_bs_nocov",
)

"""
line(
    title="Mini-batch convergence by sample size (with covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Sample size $L$",
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
    title="Mini-batch convergence by sample size (no covariance): N=%(nt)i",
    ylabel="Free energy",
    xlabel=r"Sample size $L$",
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
"""

# Numerical vs analytic approach
line(
    title="Best free energy by sample size (with covariance): N=%(nt)i",
    ylabel="Free energy",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES[1:],
    points3d="cost_history",
    cov="cov",
    lr=0.05,
    normalize_mean=True,
    savename="best_cost_ss_num_cov",
)

line(
    title="Best free energy by sample size (no covariance): N=%(nt)i",
    ylabel="Free energy",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("nt", NT),},
    lines=("num", NUM),
    points_param="ss",
    points_values=SAMPLE_SIZES[1:],
    points3d="cost_history",
    cov="nocov",
    lr=0.05,
    normalize_mean=True,
    savename="best_cost_ss_num_nocov",
)

"""
for param in ("amp1", "amp2", "r1", "r2"):
    param_conv(
        title="Convergence of " + param + " (with covariance): N=%(nt)i" ,
        ylabel="Parameter value",
        subplots={"param" : ("nt", NT),},
        samples=10,
        param=param,
        lr=0.05,
        ss=20,
        cov="cov",
        savename="conv_%s_cov" % param,
    )

    param_conv(
        title="Convergence of " + param + " (no covariance): N=%(nt)i",
        ylabel="Parameter value",
        subplots={"param" : ("nt", NT),},
        samples=10,
        param=param,
        lr=0.05,
        ss=20,
        cov="nocov",
        savename="conv_%s_nocov" % param,
    )
"""
boxplot(
    title=r"$A_1$: %(method)s",
    ylabel="Value",
    ylim=(-10, 30),
    subplots={
        "items" : [
            {
                "method" : "sVB (no covariance)",
                "cov" : "nocov",
            },
            {
                "method" : "sVB (with covariance)",
                "cov" : "cov",
            },
            {
                "method" : "aVB",
                "subdir_format" : "lfab2_nt_%(nt)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="amp1",
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    savename="amp1_method",
    showfliers=True,
)

boxplot(
    title=r"$A_2$: %(method)s",
    ylabel="Value",
    ylim=(-10, 30),
    subplots={
        "items" : [
            {
                "method" : "sVB (no covariance)",
                "cov" : "nocov",
            },
            {
                "method" : "sVB (with covariance)",
                "cov" : "cov",
            },
            {
                "method" : "aVB",
                "subdir_format" : "lfab2_nt_%(nt)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="amp2",
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    savename="amp2_method",
    showfliers=True,
)

boxplot(
    title=r"$R_1$: %(method)s",
    ylabel="Value",
    ylim=(-5, 5),
    subplots={
        "items" : [
            {
                "method" : "sVB (no covariance)",
                "cov" : "nocov",
            },
            {
                "method" : "sVB (with covariance)",
                "cov" : "cov",
            },
            {
                "method" : "aVB",
                "subdir_format" : "lfab2_nt_%(nt)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="r1",
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    savename="r1_method",
    showfliers=True,
)

boxplot(
    title=r"$R_2$: %(method)s",
    ylabel="Value",
    ylim=(-10, 30),
    subplots={
        "items" : [
            {
                "method" : "sVB (no covariance)",
                "cov" : "nocov",
            },
            {
                "method" : "sVB (with covariance)",
                "cov" : "cov",
            },
            {
                "method" : "aVB",
                "subdir_format" : "lfab2_nt_%(nt)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="r2",
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    savename="r2_method",
    showfliers=True,
)

boxplot(
    title="noise: %(method)s",
    ylabel="Value",
    ylim=(-1, 3),
    subplots={
        "items" : [
            {
                "method" : "sVB (no covariance)",
                "cov" : "nocov",
            },
            {
                "method" : "sVB (with covariance)",
                "cov" : "cov",
            },
            {
                "method" : "aVB",
                "subdir_format" : "lfab2_nt_%(nt)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="noise",
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    savename="noise_method",
    showfliers=True,
)

boxplot(
    title="%(param)s (with covariance)",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    cov="cov",
    savename="params_cov",
    showfliers=True,
)

boxplot(
    title="%(param)s (no covariance)",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("nt", NT),
    lr=0.05,
    ss=20,
    cov="nocov",
    savename="params_nocov",
    showfliers=True,
)

boxplot(
    title="%(param)s (VB with covariance)",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("nt", NT),
    cov="cov",
    savename="params_vb_cov",
    showfliers=True,
    subdir_format="lfab_nt_%(nt)i",
)

"""
# Convergence of parameters by sample size
boxplot(
    title="amp1 by sample size (with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp1 by sample size (no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp2 by sample size (with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp2 by sample size (no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r2 by sample size (with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r2 by sample size (no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
    ylim=(-10, 30),
    subplots={"param" : ("nt", NT),},
    boxes=("ss", SAMPLE_SIZES),
    param="r2",
    lr=0.1,
    bs=10,
    cov="nocov",
    savename="conv_ss_r2_nocov",
)
"""

"""

boxplot(
    title="amp1 by sample size (analytic with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp1 by sample size (numerical with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp1 by sample size (analytic no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="amp1 by sample size (numerical no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (analytic with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (numerical with covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (analytic no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
    title="r1 by sample size (numerical no covariance): N=%(nt)i",
    ylabel=r"Sample size $L$",
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
"""
if "--show" in sys.argv:
    plt.show()