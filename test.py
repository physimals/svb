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
    # set the properties of the simulated data
    true_vals = {
        "amp1" : [1.0, 0.8],
        "amp2" : [0.5, 0.7],
        "r1" : [1.0, 1.5],
        "r2" : [0.2, 0.4],
        "noise_sd" : 0.2,
    }
    true_vals["noise_var"] = true_vals["noise_sd"]**2

    # Time points - for time being, make sure we have a good range of samples based on the choice of R1
    t_end = 5*1/true_vals["r1"][0]
    t = np.linspace(0, t_end, num=N_SAMPLES)
    
    y_true, x = model.test_data(t, true_vals)
    print("data shape", y_true.shape)
    print("data max: ", np.max(x))
    print(t)
    print(y_true[0])
    print(y_true[1])

    # Train with no correlation between mean and variance
    vae_no_post_corr = VaeNormalFit(model, nt=N_SAMPLES, nvoxels=2, mode_corr='no_post_corr', learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, draw_size=DRAW_SIZE)
    _, cost_history_no_post_corr = vae_no_post_corr.train(t, x, training_epochs=TRAINING_EPOCHS)
    print(vae_no_post_corr.mp_covar_out)

    vae_post_corr = vae_no_post_corr
    print(vae_post_corr.mp_mean_out)
    print(vae_post_corr.mp_covar_out)
    
    mean_corr = vae_post_corr.mp_mean_out
    mean_nocorr = vae_no_post_corr.mp_mean_out
    sd_corr = np.sqrt(np.diag(vae_post_corr.mp_covar_out))
    sd_nocorr = np.sqrt(np.diag(vae_no_post_corr.mp_covar_out))

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

    #ax2 = plt.subplot(1,2,2)
    #plt.cla()
    #plt.plot(cost_history)
    #plt.ylabel('cost')
    #plt.xlabel('epochs')
    #plt.title('Infer post correlation')

    plt.show()

def test(model):
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
    print("data max: ", np.max(x))

    # Train with no correlation between mean and variance
    vae_no_post_corr = VaeNormalFit(model, nt=N_SAMPLES, nvoxels=1, mode_corr='no_post_corr', learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, draw_size=DRAW_SIZE)
    _, cost_history_no_post_corr = vae_no_post_corr.train(t, x, training_epochs=TRAINING_EPOCHS)
    print(vae_no_post_corr.mp_covar_out)

    # now train with correlation between mean and variance using previous run as initialization
    vae_post_corr = VaeNormalFit(model, nt=N_SAMPLES, nvoxels=1, mode_corr='infer_post_corr', learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, draw_size=DRAW_SIZE, vae_init=vae_no_post_corr)
    _, cost_history = vae_post_corr.train(t, x, training_epochs=TRAINING_EPOCHS)

    mean_corr = vae_post_corr.mp_mean_out[0]
    mean_nocorr = vae_no_post_corr.mp_mean_out[0]
    sd_corr = np.sqrt(np.diagonal(vae_post_corr.mp_covar_out, 1, 2))[0]
    sd_nocorr = np.sqrt(np.diagonal(vae_no_post_corr.mp_covar_out, 1, 2))[0]

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
    plt.figure(4)
    #y_est_init = model.ievaluate(vae_init.mp_mean_out, t)
    y_est_no_post_corr = model.ievaluate(vae_no_post_corr.mp_mean_out, t)
    y_est_post_corr = model.ievaluate(vae_post_corr.mp_mean_out, t)

    # Output without post corr
    ax1 = plt.subplot(1,2,1)
    plt.cla()
    # Noisy data (red x)
    plt.plot(t,x,'rx')
    # Ground truth (red line)
    plt.plot(t, y_true,'r')
    # Initial guess (black)
    #plt.plot(t, y_est_init, 'k.')
    # Fit with estimated parameter values (blue)
    plt.plot(t, y_est_no_post_corr, 'b')

    # Same for post corr
    ax2 = plt.subplot(1,2,2)
    plt.cla()
    plt.plot(t,x,'rx')
    plt.plot(t,y_true,'r')
    #plt.plot(t, y_est_init,'k.')
    plt.plot(t,y_est_post_corr,'b')
    plt.show()

if __name__ == "__main__":
    tf.set_random_seed(1)
    #test(BiExpModel({}))
    test(ExpModel({}))
    #test_multivox(ExpModel({}))