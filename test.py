import os

import nibabel as nib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import ExpModel, BiExpModel
from inference import VaeNormalFit

N_SAMPLES = 100
LEARNING_RATE = 0.02
BATCH_SIZE = 10
TRAINING_EPOCHS = 100
DRAW_SIZE = BATCH_SIZE

def test_multivox(model):
    NUM_VOX = 10

    # set the properties of the simulated data
    true_vals = {
        "amp1" : [1.0] * NUM_VOX,
        "amp2" : [0.5] * NUM_VOX,
        "r1" : [1.0] * NUM_VOX,
        "r2" : [0.2] * NUM_VOX,
        "noise_sd" : 0.2,
    }
    true_vals["noise_var"] = true_vals["noise_sd"]**2

    # Time points - for time being, make sure we have a good range of samples based on the choice of R1
    t_end = 5*1/true_vals["r1"][0]
    t = np.linspace(0, t_end, num=N_SAMPLES)
    
    y_true, x = model.test_data(t, true_vals)

    # Uncomment to switch to one-vox but same data
    #NUM_VOX = 1
    #y_true = y_true[0, :].reshape(1, -1)
    #x = x[0, :].reshape(1, -1)

    # Train with no correlation between mean and variance
    vae_no_post_corr = VaeNormalFit(model, 
                                    mode_corr='no_post_corr',
                                    learning_rate=LEARNING_RATE,
                                    draw_size=DRAW_SIZE)
    _, cost_history_no_post_corr = vae_no_post_corr.train(t, x, batch_size=BATCH_SIZE, training_epochs=TRAINING_EPOCHS)
    
    # Train with correlation between mean and variance
    vae_post_corr = VaeNormalFit(model, nt=N_SAMPLES, nvoxels=NUM_VOX, 
                                 mode_corr='infer_post_corr', 
                                 learning_rate=LEARNING_RATE, 
                                 draw_size=DRAW_SIZE)
    _, cost_history_post_corr = vae_post_corr.train(t, x, batch_size=BATCH_SIZE, training_epochs=TRAINING_EPOCHS)
    
    # Output mean values of parameters across voxels
    mean_init = vae_post_corr.mp_mean_1[0]
    mean_corr = np.mean(vae_post_corr.output("mp_mean"), axis=0)
    mean_nocorr = np.mean(vae_no_post_corr.output("mp_mean"), axis=0)
    sd_corr = np.sqrt(np.diagonal(np.mean(vae_post_corr.output("mp_covar"), axis=0)))
    sd_nocorr = np.sqrt(np.diagonal(np.mean(vae_no_post_corr.output("mp_covar"), axis=0)))

    for idx, param in enumerate(model.params):
        print("Parameter: %s\tInput: %f\tOutput (corr): %f (%f)\tOutput (no corr): %f (%f)" % (param.name, true_vals[param.name][0], mean_corr[idx], sd_corr[idx], mean_nocorr[idx], sd_nocorr[idx]))

    print("Noise Variance: Input: %f\tOutput (corr): %f\tOutput (no corr): %f" % (true_vals["noise_var"], np.exp(mean_corr[-1]), np.exp(mean_nocorr[-1])))

    # Plot cost history
    plt.figure(3)
    plt.subplots_adjust(hspace=2)
    plt.subplots_adjust(wspace=0.8)

    ax1 = plt.subplot(1,2,1)
    plt.cla()
    plt.plot(cost_history_no_post_corr)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('No post correlation')

    ax2 = plt.subplot(1,2,2)
    plt.cla()
    plt.plot(cost_history_post_corr)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Infer post correlation')

    # If not too many voxels, display fit at each voxel
    if NUM_VOX <= 10:
        for vox in range(x.shape[0]):
            mean_corr = vae_post_corr.output("mp_mean")[vox]
            mean_nocorr = vae_no_post_corr.output("mp_mean")[vox]
            sd_corr = np.sqrt(np.diagonal(vae_post_corr.output("mp_covar")[vox]))
            sd_nocorr = np.sqrt(np.diagonal(vae_no_post_corr.output("mp_covar")[vox]))

            print("\nVoxel %i" % vox)
            for idx, param in enumerate(model.params):
                print("Parameter: %s\tInput: %f\tOutput (corr): %f (%f)\tOutput (no corr): %f (%f)" % (param.name, true_vals[param.name][vox], mean_corr[idx], sd_corr[idx], mean_nocorr[idx], sd_nocorr[idx]))

            print("Noise Variance: Input: %f\tOutput (corr): %f\tOutput (no corr): %f" % (true_vals["noise_var"], np.exp(mean_corr[-1]), np.exp(mean_nocorr[-1])))

            plt.figure(4+vox)
            y_est_init = model.ievaluate(mean_init, t)
            y_est_no_post_corr = model.ievaluate(mean_corr, t)
            y_est_post_corr = model.ievaluate(mean_nocorr, t)

            # Output without post corr
            # Noisy data (red x)
            # Ground truth (green)
            # Initial guess (black)
            # Fit with estimated parameter values (blue)
            ax1 = plt.subplot(1,2,1)
            plt.cla()
            plt.plot(t, x[vox],'rx')
            plt.plot(t, y_true[vox],'g')
            plt.plot(t, y_est_init, 'k')
            plt.plot(t, y_est_no_post_corr, 'b')

            # Same for post corr
            ax2 = plt.subplot(1,2,2)
            plt.cla()
            plt.plot(t, x[vox],'rx')
            plt.plot(t, y_true[vox],'g')
            plt.plot(t, y_est_init,'k')
            plt.plot(t, y_est_post_corr,'b')

    plt.show()

def test_onevox(model):
    # set the properties of the simulated data
    true_vals = {
        "amp1" : 1.0,
        "amp2" : 0.5,
        "r1" : 1.0,
        "r2" : 0.2,
        "noise_sd" : 0.2
    }
    true_vals["noise_var"] = true_vals["noise_sd"]**2
        
    # Time points - for time being, make sure we have a good range of samples based on the choice of R1
    t_end = 5*1/true_vals["r1"]
    t = np.linspace(0, t_end, num=N_SAMPLES)
    
    y_true, x = model.test_data(t, true_vals)

    # Train with no correlation between mean and variance
    vae_no_post_corr = VaeNormalFit(model, mode_corr='no_post_corr', learning_rate=LEARNING_RATE, draw_size=DRAW_SIZE)
    _, cost_history_no_post_corr = vae_no_post_corr.train(t, x, batch_size=BATCH_SIZE, training_epochs=TRAINING_EPOCHS)

    # now train with correlation between mean and variance using previous run as initialization
    vae_post_corr = VaeNormalFit(model, mode_corr='infer_post_corr', learning_rate=LEARNING_RATE, draw_size=DRAW_SIZE)
    _, cost_history = vae_post_corr.train(t, x, batch_size=BATCH_SIZE, training_epochs=TRAINING_EPOCHS)

    mean_init = vae_post_corr.mp_mean_1[0]
    mean_corr = vae_post_corr.output("mp_mean")[0]
    mean_nocorr = vae_no_post_corr.output("mp_mean")[0]
    sd_corr = np.sqrt(np.diagonal(vae_post_corr.output("mp_covar")[0]))
    sd_nocorr = np.sqrt(np.diagonal(vae_no_post_corr.output("mp_covar")[0]))

    for idx, param in enumerate(model.params):
        print("Parameter: %s\tInput: %f\tOutput (corr): %f (%f)\tOutput (no corr): %f (%f)" % (param.name, true_vals[param.name], mean_corr[idx], sd_corr[idx], mean_nocorr[idx], sd_nocorr[idx]))

    print("Noise Variance: Input: %f\tOutput (corr): %f\tOutput (no corr): %f" % (true_vals["noise_var"], np.exp(mean_corr[-1]), np.exp(mean_nocorr[-1])))

    # plot cost history
    plt.figure(3)
    plt.subplots_adjust(hspace=2)
    plt.subplots_adjust(wspace=0.8)

    ax1 = plt.subplot(1,2,1)
    plt.cla()
    plt.plot(cost_history_no_post_corr)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('No post correlation')

    ax2 = plt.subplot(1,2,2)
    plt.cla()
    plt.plot(cost_history)
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title('Infer post correlation')

    # plot the estimated functions overlaid on data
    # Noisy data (red x)
    # Ground truth (green line)
    # Initial guess (black)
    # Fit with estimated parameter values (blue)
    plt.figure(4)
    y_est_init = model.ievaluate(mean_init, t)
    y_est_no_post_corr = model.ievaluate(mean_nocorr, t)
    y_est_post_corr = model.ievaluate(mean_corr, t)

    # Output without post corr
    ax1 = plt.subplot(1,2,1)
    plt.cla()
    plt.plot(t,x[0],'rx')
    plt.plot(t, y_true[0],'g')
    plt.plot(t, y_est_init, 'k')
    plt.plot(t, y_est_no_post_corr, 'b')

    # Same for post corr
    ax2 = plt.subplot(1,2,2)
    plt.cla()
    plt.plot(t,x[0],'rx')
    plt.plot(t,y_true[0],'g')
    plt.plot(t, y_est_init,'k')
    plt.plot(t,y_est_post_corr,'b')
    plt.show()

def test_nifti(fname, model, t0, dt, outdir="."):
    """
    Fit to a 4D Nifti image

    :param fname: File name of Nifti image
    :param mode: Model to fit to
    :param t0: Timeseries first value
    :param dt: Time step
    """
    nii = nib.load(fname)
    d = nii.get_data()
    shape = d.shape
    d_flat = d.reshape(-1, shape[-1])

    # Generate timeseries FIXME should be derivable from the model
    t = np.linspace(t0, t0+shape[3]*dt, num=shape[3], endpoint=False)

    # Train with no correlation between parameters
    vae = VaeNormalFit(model, mode_corr='no_post_corr',
                       learning_rate=LEARNING_RATE,
                       draw_size=DRAW_SIZE)
    _, cost_history_no_corr = vae.train(t, d_flat, batch_size=BATCH_SIZE, training_epochs=TRAINING_EPOCHS)
    
    # Transpose output so the first index is by parameter
    means = np.transpose(vae.output("mp_mean"))
    variances = np.transpose(np.diagonal(vae.output("mp_covar"), axis1=1, axis2=2))

    # Write out parameter mean and variance images
    makedirs(outdir, exist_ok=True)
    for idx, param in enumerate(model.params):
        nii_mean = nib.Nifti1Image(means[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_mean.to_filename(os.path.join(outdir, "mean_%s.nii.gz" % param.name))
        nii_var = nib.Nifti1Image(variances[idx].reshape(shape[:3]), None, header=nii.get_header())
        nii_var.to_filename(os.path.join(outdir, "var_%s.nii.gz" % param.name))

    # Write out modelfit
    # FIXME currently have to tile parameters because require 1 value per time point in evaluate
    tiled_params = np.tile(np.expand_dims(means, axis=-1), (1, 1, len(t)))
    fit = model.ievaluate(tiled_params, t)
    fit_nii = nib.Nifti1Image(fit.reshape(shape), None, header=nii.get_header())
    fit_nii.to_filename(os.path.join(outdir, "modelfit.nii.gz"))

def makedirs(d, exist_ok=False):
    try:
        os.makedirs(d)
    except OSError as e:
        import errno
        if not exist_ok or e.errno != errno.EEXIST:
            raise

if __name__ == "__main__":
    # To make tests repeatable
    tf.set_random_seed(1)
    np.random.seed(1)

    #test_onevox(BiExpModel({}))
    #test_multivox(BiExpModel({}))
    test_nifti("test_data_exp.nii", BiExpModel({}), 0, 0.02, "noisy")

