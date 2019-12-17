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
BATCH_SIZES = (5, 6, 9, 12, 18, 24, 48)
SAMPLE_SIZES = (2, 5, 10, 20)
DECAY_RATES = (1.0, 0.5, 0.1)
RPTS = (1, 8)
NUM = ("num", "analytic")
COV = ("cov", "nocov")

#BASEDIR = "/mnt/hgfs/win/data/svb/asl"
#MASK = "/mnt/hgfs/win/data/asl/fsl_course/mpld_asltc_mask.nii.gz"
BASEDIR = "c:/Users/ctsu0221/dev/data/svb/asl3"
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

LABELS = {
    "ss" : r"Sample size $L$",
    "bs" : r"Batch size $B$",
    "lr" : r"$\alpha$",
    "ftiss" : r"$f_{tiss}$",
    "delttiss" : r"$\delta t$",
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
    subdir_format = params.get("subdir", "rpts_%(rpts)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s")
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
    nx, ny, items = subplot_info(subplots)
    plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)

        # Full repeat data only done on sample sizes up to 20
        #if params["rpts"] == 8 and "ss" in params:
        #    params["ss"] = min(params["ss"], 20) 
            
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

            data.append(nib.load(os.path.join(subdir(params), "mean_%s.nii.gz" % param)).get_data()[MASKDATA != 0])
            labels.append(label)

        plt.boxplot(data, labels=labels, showfliers=kwargs.get("showfliers", False), showmeans=True)
        title = kwargs.get("title", "") % params
        plt.title(title)
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
        if params.get("rpts", 8) == 8 and "ss" in params:
            params["ss"] = min(params["ss"], 20) 

        line_param = lines[0]
        for line_val in lines[1]:
            params = dict(params)
            params.update(kwargs)
            params[line_param] = line_val

            if isinstance(line_val, float):
                label = "%s=%.3f" % (LABELS[line_param], line_val)
            else:
                label = "%s=%s" % (LABELS[line_param], str(line_val))

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
            params[kwargs["points_param"]] = point_val
            if subdir(params) is None:
                mean_cost_histories.append(None)
                continue

            cost_history = nib.load(os.path.join(subdir(params), "cost_history.nii.gz")).get_data()
            mean_cost_history = np.mean(cost_history, axis=(0, 1, 2))       
            mean_cost_histories.append(mean_cost_history)

            best_cost = mean_cost_history[-1]
            #print(subdir(params), best_cost)
            #print(mean_cost_history)
            if best_cost < very_best_cost:
                very_best_cost = best_cost
            
        #print("Very best FE=", very_best_cost)
        for percentage_factor in (1.01, 1.02, 1.05, 1.1, 1.2):
            label = "Within %i%% of best FE" % int((percentage_factor - 1)*100+0.4)
            points = []
           
            for idx, point_val in enumerate(kwargs["points_values"]):
                params[kwargs["points_param"]] = point_val
                if subdir(params) is None:
                    points.append(None)
                    continue

                mean_cost_history = mean_cost_histories[idx]
                best_cost = mean_cost_history[-1]
                #print(subdir(params), best_cost)
                #print(mean_cost_history)
                #converged_cost = best_cost * percentage_factor
                converged_cost = very_best_cost * percentage_factor
                #print(converged_cost)
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
                
            fmt=''
            if len([p for p in points if p is not None]) < 2:
                fmt = 'o'
            plt.plot(kwargs["points_values"], points, fmt, label=label)

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

def images(subplots, **kwargs):
    nx, ny, items = subplot_info(subplots)
    fig = plt.figure(figsize=kwargs.get("figsize", (5*nx, 5*ny)))

    for idx, params in enumerate(items):
        plt.subplot(ny, nx, idx+1)
        params = dict(params)
        params.update(kwargs)

        param = kwargs["param"]
        slice_z = kwargs["slice_z"]
        print(params)
        print(subdir(params))
        data = nib.load(os.path.join(subdir(params), "mean_%s.nii.gz") % param).get_data()
        slice_data = data[:, :, slice_z]
        slice_data *= kwargs.get("scale", 1.0)
        im = plt.imshow(slice_data.transpose(), **kwargs.get("pltargs", {}))
        
        plt.xticks([])
        plt.yticks([])
        plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        title = kwargs.get("title", "") % params
        plt.title(title)
    
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)

    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.83, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label(kwargs.get("cbar_label", ""), rotation=90)

    #set the colorbar ticks and tick labels
    #cbar.set_ticks(np.arange(0, 1.1, 0.5))

    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done image plot: %s" % title)

def timeseries(**kwargs):
    svb_subdir = subdir(kwargs)
    print("timeseries", svb_subdir)
    fab_subdir = os.path.join(BASEDIR, "fab_rpts_8")

    PLDS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    pos = kwargs["pos"]
    src_data = nib.load(os.path.join(BASEDIR, "mpld_asltc_diff.nii.gz")).get_data()
    mean_data = nib.load(os.path.join(BASEDIR, "mpld_asltc_diff_mean.nii.gz")).get_data()
    svb_data = nib.load(os.path.join(svb_subdir, "modelfit.nii.gz")).get_data()
    fab_data = nib.load(os.path.join(fab_subdir, "modelfit.nii.gz")).get_data()
    src_y = src_data[pos[0], pos[1], pos[2], :]
    mean_y = mean_data[pos[0], pos[1], pos[2], :]
    svb_y = svb_data[pos[0], pos[1], pos[2], :][::8]
    fab_y = fab_data[pos[0], pos[1], pos[2], :][::8]
    src_x = []
    for pld in PLDS:
        src_x += [pld,] * 8
    plt.plot(src_x, src_y, "o", label="Source data")
    plt.plot(PLDS, mean_y, "v-", label="Mean source data")
    plt.plot(PLDS, svb_y, "o-", label="sVB")
    plt.plot(PLDS, fab_y, "x-", label="aVB")
        
    title = kwargs.get("title", "") % kwargs
    plt.title(title)
    plt.ylabel("Signal")
    plt.xticks(PLDS)
    plt.xlabel("Post-labelling delay (s)")
    plt.legend()
    plt.savefig("%s.png" % kwargs.get("savename", "fig"))
    print("Done timeseries plot: %s" % title)

timeseries(
    title="Timeseries and model fit for ASL MRI data",
    lr=0.1,
    ss=20,
    pos=(23, 16, 13),
    cov="cov",
    rpts=8,
    savename="timeseries",
)
 
images(
    title="Perfusion (%(method)s - RPTS=%(rpts)i)",
    subplots={"items" : [
        #{
        #    "subdir" : "rpts_%(rpts)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s",
        #    "method" : "sVB",
        #    "rpts" : 1,
        #},
        #{
        #    "subdir" : "fab_rpts_%(rpts)i",
        #    "method" : "aVB",
        #    "rpts" : 1,
        #},
        {            
            "subdir" : "rpts_%(rpts)i_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_%(num)s_%(cov)s",
            "method" : "sVB",
            "rpts" : 8,
            "ss" : 5,
        },
        {
            "subdir" : "fab_rpts_%(rpts)i",
            "method" : "aVB",
            "rpts" : 8,
            "ss" : 5,
        }
    ]},
    param="ftiss",
    slice_z=13,
    lr=0.1,
    ss=20,
    cov="cov",
    scale=6000 / 1313.546793 / 0.85,
    cbar_label = "ml/100g/min",
    savename="perfusion_compare",
    pltargs={
        "vmin" : 0, 
        "vmax" : 100,
        "cmap" : "hot",
    }
)

plt.show()
import sys
sys.exit(1)

# Best FE by learning rate for various sample sizes
line(
    title="Best FE by learning rate (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_ss_cov",
)

line(
    title="Best FE by learning rate (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_ss_nocov",
)

# Convergence speed by sample size
conv_speed(
    title="Convergence by sample size (with covariance): RPTS=%(rpts)i",
    ylabel=r"Time to convergence ($s$)",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("rpts", RPTS),},
    points_param="ss",
    points_values=SAMPLE_SIZES,
    lr=0.1,
    cov="cov",
    savename="conv_speed_ss_cov",
)

conv_speed(
    title="Convergence by sample size (no covariance): RPTS=%(rpts)i",
    ylabel=r"Time to convergence ($s$)",
    xlabel=r"Sample size $L$",
    subplots={"param" : ("rpts", RPTS),},
    points_param="ss",
    points_values=SAMPLE_SIZES,
    lr=0.1,
    cov="nocov",
    savename="conv_speed_ss_nocov",
)

# Best FE by learning rate for various batch sizes
line(
    title="Best FE by batch size (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    ss=5,
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="cov",
    savename="best_cost_lr_bs_cov",
)

line(
    title="Best FE by batch size (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    #ylim=[(10, 40), (20, 60), (30, 100), (60, 150)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    ss=5,
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_bs_nocov",
)

# Convergence speed by batch size
conv_speed(
    title="Convergence by batch size (with covariance): RPTS=%(rpts)i",
    ylabel=r"Time to convergence ($s$)",
    xlabel="Batch size",
    subplots={"param" : ("rpts", RPTS),},
    points_param="bs",
    ss=5,
    points_values=BATCH_SIZES,
    lr=0.1,
    cov="cov",
    savename="conv_speed_bs_cov",
)

conv_speed(
    title="Convergence by batch size (no covariance): RPTS=%(rpts)i",
    ylabel=r"Time to convergence ($s$)",
    xlabel="Batch size",
    subplots={"param" : ("rpts", RPTS),},
    points_param="bs",
    ss=5,
    points_values=BATCH_SIZES,
    lr=0.1,
    cov="nocov",
    savename="conv_speed_bs_nocov",
)

# Parameters
boxplot(
    title=r"$f_{tiss}$: %(method)s",
    ylabel="Value",
    ylim=(-5, 25),
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
                "subdir_format" : "fab_rpts_%(rpts)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="ftiss",
    boxes=("rpts", RPTS),
    lr=0.1,
    ss=20,
    savename="ftiss_method",
    showfliers=False,
)

boxplot(
    title=r"$\delta t$: %(method)s",
    ylabel="Value",
    ylim=(-1, 5),
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
                "subdir_format" : "fab_rpts_%(rpts)i",
            },
        ],
        "layout" : (3, 1),
    },
    figsize=(10, 5),
    param="delttiss",
    boxes=("rpts", RPTS),
    lr=0.1,
    ss=20,
    savename="delttiss_method",
    showfliers=False,
)

boxplot(
    title="%(param)s (with covariance)",
    ylabel="Value",
    ylim=[(-5, 25), (-1, 5),],
    subplots={"param" : ("param", ("ftiss", "delttiss", )),},
    boxes=("rpts", RPTS),
    lr=0.1,
    ss=5,
    cov="cov",
    savename="params_cov",
)

boxplot(
    title="%(param)s (no covariance)",
    ylabel="Value",
    ylim=[(-5, 25), (-1, 5),],
    subplots={"param" : ("param", ("ftiss", "delttiss", )),},
    boxes=("rpts", RPTS),
    lr=0.1,
    ss=5,
    cov="nocov",
    savename="params_nocov",
)

"""
# Learning rate decay
line(
    title="Convergence of FE with learning rate decay (lr=%(lr).3f)",
    ylabel="Free energy",
    ylim=(77, 82),
    subplots={"param" : ("lr", LEARNING_RATES),},
    lines=("dr", DECAY_RATES),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    savename="conv_lr_cov",
    subdir="qtests4_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_dr_%(dr).3f",
    rpts=8,
    ss=10,
    bs=9,
)

# Learning rate decay
line(
    title="Convergence of FE with learning rate decay rate %(dr).3f",
    ylabel="Free energy",
    ylim=(77, 82),
    subplots={"param" : ("dr", DECAY_RATES),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    savename="conv_lr_cov",
    subdir="qtests4_lr_%(lr).3f_bs_%(bs)i_ss_%(ss)i_dr_%(dr).3f",
    rpts=8,
    ss=10,
    bs=9,
)
"""

"""
# Convergence of FE by learning rate
line(
    title="Convergence of FE (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="cov",
    savename="conv_lr_cov",
)

line(
    title="Convergence of FE (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("lr", LEARNING_RATES),
    points4d="cost_history",
    runtime=True,
    cov="nocov",
    savename="conv_lr_nocov",
)

# Best FE by learning rate for various sample sizes
line(
    title="Best FE by learning rate (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
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
    title="Best FE by learning rate (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("ss", SAMPLE_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_ss_nocov",
)

# Convergence of FE by batch size
line(
    title="Convergence of FE by batch size (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="cov",
    savename="conv_bs_cov",
)

line(
    title="Convergence of FE by batch size (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points4d="cost_history",
    runtime=True,
    lr=0.05,
    cov="nocov",
    savename="conv_bs_nocov",
)

# Best FE by learning rate for various batch sizes
line(
    title="Best FE by batch size (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    xlabel=r"Learning rate $\alpha$",
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
    title="Best FE by batch size (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    #ylim=(20, 100),
    subplots={"param" : ("rpts", RPTS),},
    lines=("bs", BATCH_SIZES),
    points_param="lr",
    points_values=LEARNING_RATES,
    points3d="cost_history",
    cov="nocov",
    savename="best_cost_lr_bs_nocov",
)

# Convergence of FE by sample size
line(
    title="Convergence by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
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
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
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
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
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
    ylabel="Free energy",
    ylim=[(10, 20), (75, 95)],
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
    ylabel=r"Sample size $L$",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_ftiss_cov",
)

boxplot(
    title="ftiss by sample size (no covariance): RPTS=%(rpts)i",
    ylabel=r"Sample size $L$",
    ylim=(-10, 30),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="ftiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_ftiss_nocov",
)

boxplot(
    title="delttiss by sample size (with covariance): RPTS=%(rpts)i",
    ylabel=r"Sample size $L$",
    ylim=(-1, 4),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="delttiss",
    lr=0.1,
    bs=6,
    cov="cov",
    savename="conv_ss_delttiss_cov",
)

boxplot(
    title="delttiss by sample size (no covariance): RPTS=%(rpts)i",
    ylabel=r"Sample size $L$",
    ylim=(-1, 4),
    subplots={"param" : ("rpts", RPTS),},
    boxes=("ss", SAMPLE_SIZES),
    param="delttiss",
    lr=0.1,
    bs=6,
    cov="nocov",
    savename="conv_ss_delttiss_nocov",
)
"""
"""
boxplot(
    title="noise by sample size (with covariance): RPTS=%(rpts)i",
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
"""
# Numerical vs analytic approach
line(
    title="Best FE by sample size (with covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
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
    title="Best FE by sample size (no covariance): RPTS=%(rpts)i",
    ylabel="Free energy",
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
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
"""
boxplot(
    title="noise by sample size (analytic with covariance): RPTS=%(rpts)i",
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
    ylabel=r"Sample size $L$",
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
