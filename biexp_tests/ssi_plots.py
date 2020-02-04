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
SAMPLE_SIZE_FACTORS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)
INITIAL_SAMPLE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FINAL_SAMPLE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

NOISE = (1.0, 2.0, 5.0, 10.0)
NT = (10, 20, 50, 100)
NUM = ("num", "analytic")
COV = ("cov", "nocov")
BASEDIR = "c:/Users/ctsu0221/dev/data/svb/biexp"
#BASEDIR = "/mnt/hgfs/win/data/svb/biexp"

DEFAULTS = {
    "lr" : 0.1,
    "bs" : None,
    "ssi" : 200,
    "ssf" : 1.0,
    "nt" : 100,
    "num" : "analytic",
    "cov" : "cov",
    "noise" : 1.0,
}

LABELS = {
    "ssf" : r"Final Sample size",
    "ssi" : r"Initial sample size",
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

def get_subdir(params):
    """
    :return: Output subdirectory corresponding to set of parameters
    """
    fullparams = dict(DEFAULTS)
    fullparams.update(params)
    if fullparams["bs"] is None:
        # If batch size unspecified assume no mini-batch optimization
        fullparams["bs"] = fullparams["nt"]
    subdir_format = params.get("subdir_format", "nt_%(nt)i_noise_%(noise).1f_lr_%(lr).3f_bs_%(bs)i_ssi_%(ssi)i_ssf_%(ssf)i")
    subdir = os.path.join(BASEDIR, subdir_format % fullparams)
    if not os.path.exists(os.path.join(subdir, "runtime")):
        print("No results for: %s" % subdir)
        return None
    else:
        return subdir

def get_subplots(subplots):
    """
    :return nx, ny, sequence of parameter dictionaries for subplots
    """
    if subplots is None:
        nx, ny, plot_params = 1, 1, [{}]
    else:
        if "items" in subplots:
            plot_params = subplots["items"]
        elif "param" in subplots:
            param, values = subplots["param"]
            plot_params = [{param : v} for v in values]

        if "layout" in subplots:
            nx, ny = subplots["layout"]
        else:
            nx = ny = int(math.sqrt(len(plot_params)))
            while nx*ny < len(plot_params):
                if nx <= ny:
                    nx += 1
                else:
                    ny += 1
    return nx, ny, plot_params

def get_dataset(params, name):
    subdir = get_subdir(params)
    if subdir is None:
        return None

    fname = os.path.join(subdir, "%s.nii.gz" % name)
    if not os.path.exists(fname):
        return None
    else:
        return nib.load(fname).get_data()

def get_runtime(params, epoch=None, n_epochs=None):
    subdir = get_subdir(params)
    if subdir is None:
        return None

    fname_history = os.path.join(subdir, "runtime_history")
    fname_total = os.path.join(subdir, "runtime")
    if epoch is None or not os.path.exists(fname_history):
        with open(os.path.join(subdir, "runtime")) as runtime_file:
            total_runtime = float(runtime_file.read())
        if epoch is None:
            return total_runtime
        else:
            if n_epochs is None:
                raise ValueError("Can't get runtime - don't know how many epochs")
            print("Estimating runtime")
            return epoch * total_runtime / n_epochs
    else:
        with open(os.path.join(subdir, "runtime_history")) as runtime_file:
            runtime_history = [float(v) for v in runtime_file.readlines()]
        print("Accurate runtime", epoch, len(runtime_history))
        return runtime_history[epoch]

def boxplot(subplots, boxes, **kwargs):
    """
    Generate a box plot for values of a parameter
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, plot_params = get_subplots(subplots)

    for idx, params in enumerate(plot_params):
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

            if get_subdir(params) is None:
                continue

            labels.append(label)
            data_file = os.path.join(get_subdir(params), "mean_%s.nii.gz" % param)
            if os.path.exists(data_file):
                param_data = nib.load(data_file).get_data().flatten()
                if param == "noise" and "fab" not in get_subdir(params):
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

def line(subplots, **kwargs):
    """
    Generate line plots
    """
    plt.figure(figsize=kwargs.get("figsize", (10, 10)))
    nx, ny, plot_params = get_subplots(subplots)

    for idx, subplot_params in enumerate(plot_params):
        plt.subplot(ny, nx, idx+1)

        line_param, line_vals = kwargs.get("lines",  ("", [1]))
        for line_val in line_vals:
            params = dict(subplot_params)
            params.update(kwargs)
            
            params[line_param] = line_val
            if isinstance(line_val, float):
                line_label = "%s=%.3f" % (LABELS[line_param], line_val)
            else:
                line_label = "%s=%s" % (LABELS[line_param], str(line_val))

            if "timeseries" in kwargs:
                timeseries_data = get_dataset(params, kwargs["timeseries"])
                if timeseries_data is None:
                    continue
                points = np.mean(timeseries_data, axis=(0, 1, 2))
            elif "points" in kwargs:
                points_param, points_values, dataset = kwargs["points"]
                points = []
                for point_val in points_values:
                    params[points_param] = point_val
                    data = get_dataset(params, dataset)
                    if data is None:
                        points.append(None)
                        continue
                    
                    if data.ndim == 4:
                        # For 4D data take the last volume
                        data = data[..., -1]
                    points.append(np.mean(data))
                plt.xticks(range(len(points_values)), ["%.2f" % v for v in points_values])
            else:
                raise RuntimeError("No points to plot")

            if kwargs.get("runtime", None):
                with open(os.path.join(subdir, "runtime")) as runtime_file:
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

            plt.plot([float(v) / xscale for v in range(len(points))], points, label=line_label)

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
                
        if len(line_vals) > 1:
            plt.legend()

    plt.tight_layout()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done line plot: %s" % title)

def conv_speed(subplots, **kwargs):
    nx, ny, plot_params = get_subplots(subplots)
    plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, subplot_params in enumerate(plot_params):
        plt.subplot(ny, nx, idx+1)
        params = dict(subplot_params)
        params.update(kwargs)
    
        very_best_cost = 1e99
        mean_cost_histories = []
        points_param, points_values = kwargs["points"]
        
        for point_val in points_values:
            params[points_param] = point_val
            subdir = get_subdir(params)
            if subdir is None:
                mean_cost_histories.append(None)
                continue

            cost_history = nib.load(os.path.join(subdir, "cost_history.nii.gz")).get_data()
            mean_cost_history = np.mean(cost_history, axis=(0, 1, 2))
            mean_cost_histories.append(mean_cost_history)

            best_cost = mean_cost_history[-1]
            if best_cost < very_best_cost:
                very_best_cost = best_cost

        for percentage_factor in (1.02, 1.05, 1.1, 1.2):
            label = "Within %i%% of best free energy" % int((percentage_factor - 1)*100+0.4)
            points = []
            converged_cost = very_best_cost * percentage_factor
           
            for idx, point_val in enumerate(points_values):
                params[points_param] = point_val
                if get_subdir(params) is None:
                    points.append(None)
                    continue

                mean_cost_history = mean_cost_histories[idx]
                best_cost = mean_cost_history[-1]
                if mean_cost_history[-1] > converged_cost:
                    # Did not converge to within this limit
                    points.append(None)
                else:
                    converged = mean_cost_history < converged_cost
                    epoch_converged = np.argmax(converged)
                    time_converged = get_runtime(params, epoch_converged, len(mean_cost_history))                    
                    points.append(time_converged)
                
            plt.plot(points_values, points, label=label)

        title = kwargs.get("title", "") % params
        plt.title(title)
        plt.ylabel(kwargs.get("ylabel", ""))
        plt.xscale('log')
        plt.xticks(points_values)
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
    nx, ny, plot_params = get_subplots(subplots)

    for idx, params in enumerate(plot_params):
        plt.subplot(ny, nx, idx+1)

        params = dict(params)
        params.update(kwargs)

        if get_subdir(params) is None:
            continue
        data = nib.load(os.path.join(get_subdir(params), "mean_%s_history.nii.gz" % param)).get_data()
        nepochs = data.shape[-1]
        data = data.reshape(-1, nepochs)

        #with open(os.path.join(get_subdir(params), "runtime")) as runtime_file:
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

# Best free energy 
line(
    title="Best free energy: N=%(nt)i, noise=%(noise).1f",
    ylabel="Free energy",
    xlabel=r"Final sample size",
    subplots={"param" : ("nt", NT),},
    lines=("ssi", INITIAL_SAMPLE_SIZES),
    points=("ssf", FINAL_SAMPLE_SIZES, "cost_history"),
    lr=0.05,
    cov="cov",
    noise=1.0,
    bs=10,
    savename="best_cost_ssi",
)

# Best free energy 
line(
    title="Best free energy: N=%(nt)i, noise=%(noise).1f",
    ylabel="Free energy",
    xlabel=r"Initial sample size",
    subplots={"param" : ("nt", NT),},
    lines=("ssf", FINAL_SAMPLE_SIZES),
    points=("ssi", INITIAL_SAMPLE_SIZES, "cost_history"),
    lr=0.05,
    cov="cov",
    noise=1.0,
    bs=10,
    savename="best_cost_ssi",
)

# Convergence speed by sample size and SNR
conv_speed(
    title="Convergence: NT=%(nt)i, final sample size=%(ssf)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Initial sample size $L$",
    subplots={"param" : ("nt", NT),},
    points=("ssi", INITIAL_SAMPLE_SIZES),
    ssf=16,
    lr=0.05,
    cov="cov",
    noise=1.0,
    bs=10,
    savename="conv_speed_ssi",
)

# Convergence speed by sample size and SNR
conv_speed(
    title="Convergence: NT=%(nt)i, final sample size=%(ssf)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Initial sample size $L$",
    subplots={"param" : ("nt", NT),},
    points=("ssi", INITIAL_SAMPLE_SIZES),
    ssf=32,
    lr=0.05,
    cov="cov",
    noise=1.0,
    bs=10,
    savename="conv_speed_ssi",
)

# Convergence speed by sample size and SNR
conv_speed(
    title="Convergence: NT=%(nt)i, final sample size=%(ssf)i",
    ylabel=r"Time to convergence $s$",
    xlabel=r"Initial sample size $L$",
    subplots={"param" : ("nt", NT),},
    points=("ssi", INITIAL_SAMPLE_SIZES),
    ssf=128,
    lr=0.05,
    cov="cov",
    noise=1.0,
    bs=10,
    savename="conv_speed_ssi",
)

boxplot(
    title="%(param)s: NT=%(nt)i, initial sample size=%(ssi)i",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("ssf", FINAL_SAMPLE_SIZES),
    showfliers=True,
    ssi=8,
    lr=0.05,
    cov="cov",
    nt=100,
    noise=1.0,
    bs=10,
    savename="params_ssf_100",
)

boxplot(
    title="%(param)s: NT=%(nt)i, initial sample size=%(ssi)i",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("ssf", FINAL_SAMPLE_SIZES),
    showfliers=True,
    ssi=1,
    lr=0.05,
    cov="cov",
    nt=100,
    noise=1.0,
    bs=10,
    savename="params_ssf_100",
)

boxplot(
    title="%(param)s: NT=%(nt)i, initial sample size=%(ssi)i",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("ssf", FINAL_SAMPLE_SIZES),
    showfliers=True,
    ssi=8,
    lr=0.05,
    cov="cov",
    nt=20,
    noise=1.0,
    bs=10,
    savename="params_ssf_100",
)

boxplot(
    title="%(param)s: NT=%(nt)i, initial sample size=%(ssi)i",
    ylabel="Value",
    ylim=[(-10, 30), (-10, 30), (-5, 5), (-10, 30), (-1, 3)],
    subplots={"param" : ("param", ("amp1", "amp2", "r1", "r2", "noise")),},
    boxes=("ssf", FINAL_SAMPLE_SIZES),
    showfliers=True,
    ssi=1,
    lr=0.05,
    cov="cov",
    nt=20,
    noise=1.0,
    bs=10,
    savename="params_ssf_100",
)

if "--show" in sys.argv:
    plt.show()