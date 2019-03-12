# -*- coding: utf-8 -*-
"""
Created on Tue 12 Mar 2019

@author: chappell
"""

# derived from vbrt_nonlin - for a general non-linear forward model defined on an imaging volume (i.e. multi-dimensional data)

# Note: taking dim 1 of an array as refering to 'rows' and 2 to 'columns'

#%%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

############
#%% Non-linear model objects (return a 1D array based on t)
# - exponential
class ExpModel:
    
    nparams = 2
    
    prior_means=None
    prior_vars=None
    
    def evaluate(self,mp,t):
        # unpack params
        A = mp[:,:,:,0]
        R1 = mp[:,:,:,1]
        
        y_pred = A * tf.exp(-R1 * t)
        return y_pred
    
    def init_params(self,t,x,mp_mean_init):
        init_amp = np.amax(x,axis=3)
        init_R1 = 0.5
        
        mp_mean_init[:,:,:,0]=init_amp
        mp_mean_init[:,:,:,1]=init_R1
        return mp_mean_init

# - biexponential
class BiExpModel:
    
    nparams = 4
    
    prior_means=[1,1,1,1]
    prior_vars=[10,10,10,10]
    
    def evaluate(self,mp,t):
        A1 = mp[:,:,:,0]
        R1 = mp[:,:,:,1]
        A2 = mp[:,:,:,2]
        R2 = mp[:,:,:,3]
        
        y_pred = A1 * tf.exp(-R1 * t) + A2 * tf.exp(-R2 * t)
        return y_pred
    
    def init_params(self,t,x,mp_mean_init):
        init_amp1 = 0.9*np.max(x)
        init_amp2 = 0.1*np.max(x)
        init_R1 = 0.5
        init_R2 = 0.1
        
        mp_mean_init[0,0]=init_amp1
        mp_mean_init[0,2]=init_R1
        mp_mean_init[0,1]=init_amp2
        mp_mean_init[0,3]=init_R2
        return mp_mean_init
        
        
#%% VAE obj

class VaeNormalFit(object):
    def __init__(self, nlinmod, 
                 learning_rate=0.001, batch_size=100, draw_size=100, mode_corr='infer_post_corr', do_folded_normal=0, vae_init=None, mp_mean_init=None, scale=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.draw_size = draw_size
        self.mode_corr=mode_corr
        self.n_modelparams=nlinmod.nparams +1 #the total model parameters is the non-linear model parameters plus noise parameters
        self.scale=scale
        
        self.nx = 1
        self.ny = 1
        self.nz = 1
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [nx, ny, nz, None], name='data') # x is a 4D array of imaging dimensions then samples
        
        # tf fixed model variables (in this case it is the time values)
        self.t = tf.placeholder(tf.float32, [nx,ny,nz, None], name='timepoints') 
        
        # Create model
        self._create_model(vae_init, mp_mean_init)
        
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer(nlinmod)
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.Session()
        
        self.sess.run(init)
        
    def _create_model(self, vae_init, mp_mean_init):
                        
        # generative model p(x | mu) = N(A*exp(-t*R1), var)
        # prior p(mp)~MVN(mp_mean_0,mp_covar_0)
        # where mp=model_params
        # approximating distributions: 
        # q(mp) ~ MVN(mp_mean, mp_covar)
                                        
        # -- Prior:         
        
        # Define priors for the parameters  - these are TF constants for the mean and covariance matrix of an MVN (as above)
        temp_mean_0 = np.zeros((self.nx,self.ny,self.nz,self.n_modelparams))
        temp_covar_0 = 100*np.ones((self.nx,self.ny,self.nz,self.n_modelparams))
        #if specified, bring in non-linear model parameter prior values
        if nlinmod.prior_means is not None:
            temp_mean_0[:,:,:,0:nlinmod.nparams] = nlinmod.prior_means
        if nlinmod.prior_vars is not None:
            temp_covar_0[:,:,:,0:nlinmod.nparams] = nlinmod.prior_vars
            
        self.mp_mean_0 = tf.constant(temp_mean_0, dtype=tf.float32)  
        self.mp_covar_0 = tf.linalg.diag(tf.constant(temp_covar_0, dtype=tf.float32))
        
        # -- Approximating Posterior:
        
        # Setup and initialise mp_mean (as a Variable to be optimised later)
        # This is the posterior means for the parameters in the approximating distribution
        if vae_init==None:
            if mp_mean_init is None:
                mp_mean_init=0.1*tf.random_normal((self.nx,self.ny,self.nz,self.n_modelparams),0,1, dtype=tf.float32)
            self.mp_mean = tf.Variable(mp_mean_init, dtype=tf.float32, name='mp_mean')
        else:
            self.mp_mean = tf.Variable(vae_init.sess.run(vae_init.mp_mean), dtype=tf.float32, name='mp_mean')
                
        # Setup mp_covar (as a Variable to be optimised later)
        # This is the posterior covariance matrix in the approximating distribution
        # Note that parameterising using chol ensures that mp_covar is positive def, although note that this does not reduce the number of params accordingly (need a tf.tril() func)    
                
        if self.mode_corr == 'infer_post_corr':
            # infer a full covariance matrix with on and off-diagonal elements
                           
            if 1:
                
                nn = self.n_modelparams
                #n_offdiags=int((np.square(nn)-nn)/2)
                                
                # off diag
                if vae_init==None:
                    self.off_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((self.nx,self.ny,self.nz,nn,nn),0,0.1,dtype=tf.float32), name='off_diag_chol_mp_covar')                    
                else:
                    self.off_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.off_diag_chol_mp_covar), name='off_diag_chol_mp_covar')                    
                
                #import pdb; pdb.set_trace()
                       
                # diag: this needs to setup separately to the off-diag as it needs to be positive
                if vae_init==None:
                    self.log_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((self.nx,self.ny,self.nz,self.n_modelparams,),-2,0.4, dtype=tf.float32), name='log_diag_chol_mp_covar')                    
                else:
                    self.log_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.log_diag_chol_mp_covar), name='log_diag_chol_mp_covar')                    

                diag_chol_mp_covar_mat = tf.linalg.diag(tf.exp(self.log_diag_chol_mp_covar), name='chol_mp_covar')

                # reshape to matrix
                self.chol_mp_covar = tf.add((diag_chol_mp_covar_mat),tf.matrix_band_part(self.off_diag_chol_mp_covar, -1, 0))

            else:
                if vae_init==None:
                    self.chol_mp_covar = tf.Variable(0.05+tf.diag(tf.constant(0.1,shape=(self.nx,self.ny,self.nz,self.n_modelparams,))), dtype=tf.float32, name='chol_mp_covar')                            
                else:
                    self.chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.chol_mp_covar), name='chol_mp_covar')                    
        
        else:     
            # infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            if vae_init==None:
                self.log_diag_chol_mp_covar = tf.Variable(tf.constant(-2,shape=(self.nx,self.ny,self.nz,self.n_modelparams,), dtype=tf.float32), name='log_diag_chol_mp_covar')                    
            else:
                self.log_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.log_diag_chol_mp_covar), name='log_diag_chol_mp_covar')                    
            
            self.chol_mp_covar = tf.linalg.diag(tf.exp(self.log_diag_chol_mp_covar), name='chol_mp_covar')    
        
        # form the covariance matrix from the chol decomposition
        self.mp_covar = tf.matmul(tf.linalg.transpose(self.chol_mp_covar),self.chol_mp_covar, name='mp_covar')
        
        # -- Generate sample of mp:
        # Use reparameterisation trick                
        # mp = mp_mean + chol(mp_covar)*eps
        
        eps = tf.random_normal((self.nx,self.ny,self.nz,self.n_modelparams, self.draw_size), 0, 1, dtype=tf.float32)
        ntile = np.floor_divide(self.batch_size,self.draw_size)
        eps = tf.tile(eps,[1,1,1,1,ntile])
        self.eps=eps
               
        self.mp = tf.add(tf.tile(tf.reshape(self.mp_mean,[self.nx,self.ny,self.nz,self.n_modelparams,1]),[1,1,1,1,self.batch_size]), tf.matmul(self.chol_mp_covar, eps))
                        
    
    def _create_loss_optimizer(self,nlinmod):
        # The loss is composed of two terms:
    
        # 1.) log likelihood
         
        # unpack noise parameter
        log_var = tf.reshape(self.mp[:,:,:,self.n_modelparams-1,:],(self.nx,self.ny,self.nz,self.batch_size,1)) #remebering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)

        # use a iid normal distribution
        # Calculate the loglikelihood given our supplied mp values
        y_pred = nlinmod.evaluate(self.mp,self.t)
        # This prediction currently has a different set of model parameters for each data point (i.e. it is not amortized) - due to the sampling process above
        
        reconstr_loss = tf.reduce_sum(0.5 * self.scale * ( tf.div(tf.square(tf.subtract(self.x,y_pred)), tf.exp(log_var)) + log_var + np.log( 2 * np.pi) ) ,0)
        # NOTE: scale (the relative scale of number of samples and size of batch) appears in here to get the relative scaling of log-likehood correct, even though we have dropped the constant term log(n_samples)                              
            
        self.reconstr_loss=reconstr_loss
        
        # 2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        # latent_loss = log_mvn_pdf(self.mp, self.mp_mean, mp_covar) \
        #                    - log_mvn_pdf(self.mp, self.mp_mean_0, self.mp_covar_0)
       
        # 2.) D_KL{q(mp)~ MVN(mp_mean, mp_covar), p(mp)~ MVN(mp_mean_0, mp_covar_0)}
        
        #import pdb; pdb.set_trace()
                 
        inv_mp_covar_0=tf.matrix_inverse(self.mp_covar_0)
        mn=tf.reshape(tf.subtract(self.mp_mean,self.mp_mean_0),(self.nx,self.ny,self.nz,self.n_modelparams,1)) # a 5D array so that we are compatible with the covariance matrix array
        
        latent_loss = 0.5*(tf.trace(tf.matmul(inv_mp_covar_0,self.mp_covar))+tf.matmul(tf.matmul(tf.linalg.transpose(mn),inv_mp_covar_0),mn) - self.n_modelparams + tf.log(tf.matrix_determinant(self.mp_covar_0, name='det_mp_covar_0') ) - tf.log(tf.matrix_determinant(self.mp_covar, name='det_mp_covar') ) )
        self.latent_loss = latent_loss
        
        # Add the two terms (and average over the batch)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)   
        
        #self.grads_and_vars = self.opt.compute_gradients(self.cost)
        
        # Use ADAM optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = \
            self.opt.minimize(self.cost)
            
    def _log_cosh(self, x):
        #cosh=tf.clip_by_value(0.5*tf.add(tf.exp(x),tf.exp(-x)),-1e6,1e6)
        cosh=0.5*tf.add(tf.exp(x),tf.exp(-x))
        log_cosh=tf.log(1e-6+cosh)
        return log_cosh
        
    def partial_fit(self, T, X):
        """Train model based on mini-batch of input data.       
        Return cost of mini-batch.
        """
        
        # Do the optimization (self.optimizer), but also calcuate the cost for reference (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been provided X         
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X, self.t: T})
        # Note use of feed_dict to pass the data into the placeholders created previously
        return cost

####
#%%

def train(nlinmod,t,data, mode_corr='infer_post_corr', learning_rate=0.01,
          batch_size=100, draw_size=100, training_epochs=10, display_step=1, do_folded_normal=False, vae_init=None, mp_mean_init=None):
    
    # need to determine the 'scale' between log-likihood and KL(post-prior) when using batches
    #scale = n_samples/batch_size
    scale = np.floor_divide(n_samples,batch_size)
    
    vae = VaeNormalFit(nlinmod,mode_corr=mode_corr, learning_rate=learning_rate, 
                                 batch_size=batch_size, draw_size=draw_size, scale=scale, do_folded_normal=do_folded_normal, vae_init=vae_init, mp_mean_init=mp_mean_init)
    
    cost_history=np.zeros([training_epochs]) 
    total_batch = int(n_samples / batch_size)
    #print("total_batch:", '%d' % total_batch )
    #cost_full_history =  np.zeros([training_epochs * total_batch])      
    
    # Training cycle
    for epoch in range(training_epochs): # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.

        avg_cost = 0.
        index = 0
        
         
        # Loop over all batches
        for i in range(total_batch):
            if 0:
                # batches are defined sequential datapoints
                batch_xs = data[:,:,:,index:index+batch_size]
                t_xs = t[:,:,:,index:index+batch_size]
                index = index + batch_size
            else:
                # batches are defined evenly through the samples
                batch_xs = data[:,:,:,i::scale]
                t_xs = t[:,:,:,i::scale]
                     
            # Fit training using batch data
            cost = vae.partial_fit(t_xs,batch_xs)
            #cost_full_history[(epoch-1)*total_batch + i] = cost
                        
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

            #print("i:", '%04d' % (i), \
            #      "cost=", "{:.9f}".format(cost));

            if np.isnan(cost):
                import pdb; pdb.set_trace()
            
        #print(vae.sess.run((vae.mp_mean,tf.reduce_mean(vae.mp_covar)), feed_dict={vae.x: batch_xs}))
                                              
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), \
                  "cost=", "{:.9f}".format(avg_cost))
        cost_history[epoch]=avg_cost   
        
    return vae, cost_history
  

############


################################################################################################
################################################################################################   
#%% main code

# set the properties of the simulated data
nx = 1
ny = 1
nz = 1

n_samples=100
true_amp1 = 1.0 # the true amplitude of the model
true_amp2 = 0.5
true_R1 = 1.0 # the true decay rate
true_R2 = 0.2
true_sd = 0.1 # the s.d. of the noise
true_var = true_sd*true_sd

#time points
t_end = 5*1/true_R1 #for time being, make sure we have a good range of samples based on the choice of R1
t = np.linspace(0,t_end,num=n_samples)
t = t.reshape((nx,ny,nz,-1))

############
#%% create simulated data

# calcuate forward model
#y_true = true_amp * np.exp(-t*true_R1)
# we will use the non-linear model as defined in the class (therefore need a TF session too)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

model="exp"
if model == "exp":
    nlinmod=ExpModel()
    mptrue = np.reshape(np.array((true_amp1,true_R1)),(nx,ny,nz,-1))
elif model == "biexp":
    nlinmod=BiExpModel()
    mptrue = np.reshape(np.array((true_amp1,true_amp2,true_R1,true_R2)),(nx,ny,nz,-1))
    
y_true = sess.run(nlinmod.evaluate(mptrue,t))

# add noise
x = np.random.normal(y_true,true_sd)


#%% do vae approach  


learning_rate=0.02
batch_size=10 #n_samples 
draw_size=batch_size #number of draws to make when doing reparameterization
#scale = n_samples/batch_size
training_epochs=100

# initialise params
# some roughly sensible init values from the data
mp_mean_init=np.zeros([nx,ny,nz,nlinmod.nparams+1]).astype('float32')
mp_mean_init = nlinmod.init_params(t,x,mp_mean_init)
init_var=np.var(x,axis=3)
mp_mean_init[:,:,:,nlinmod.nparams] = np.log(init_var) #NB becuase we happen to be infering the log of the variance
  
# call with no training epochs to get initialisation
mode_corr='infer_post_corr'            
vae_norm_init, cost_history = train(nlinmod,t, x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=0, batch_size=batch_size, draw_size=draw_size, mp_mean_init=mp_mean_init)

# now train with no correlation between mean and variance
mode_corr='no_post_corr'
vae_norm_no_post_corr, cost_history_no_post_corr = train(nlinmod,t, x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, draw_size=draw_size, vae_init=vae_norm_init)

# now train with correlation between mean and variance
#mode_corr='infer_post_corr'
vae_norm, cost_history = train(nlinmod,t, x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, draw_size=draw_size, vae_init=vae_norm_init)

#mn = vae_norm.sess.run(vae_norm.mp_mean)
#import pdb; pdb.set_trace()
#print("VAE Estimated mean:", "{:.9f}".format(mn[0,0]), "Estimated var=", "{:.9f}".format(np.exp(mn[0,1])))    
#print("True mean:", "{:.9f}".format(true_mean[0]), "True var=", "{:.9f}".format(true_var[0][0]))

#%% plot cost history
plt.figure(3)

ax1 = plt.subplot(1,2,1)
plt.cla()

plt.subplots_adjust(hspace=2)
plt.subplots_adjust(wspace=0.8)

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

#%% plot the estimated functions overlaid on data

plt.figure(4)

ax1 = plt.subplot(1,2,1)
plt.cla()

# plot the data
plt.plot(t[0,0,0,:],x[0,0,0,:],'rx')
# plot the ground truth
plt.plot(t[0,0,0,:],y_true[0,0,0,:],'r')
# plot the fit using the inital guess (for reference)
mn = vae_norm_init.sess.run(vae_norm_init.mp_mean)
y_est = sess.run(nlinmod.evaluate(mn[:,:,:,0:nlinmod.nparams],t))
plt.plot(t[0,0,0,:],y_est[0,0,0,:],'k.')
# plto the fit with the estimated parameter values
mn = vae_norm_no_post_corr.sess.run(vae_norm_no_post_corr.mp_mean)
y_est = sess.run(nlinmod.evaluate(mn[:,:,:,0:nlinmod.nparams],t))
plt.plot(t[0,0,0,:],y_est[0,0,0,:],'b')

ax2 = plt.subplot(1,2,2)
plt.cla()

# plot the data
plt.plot(t[0,0,0,:],x[0,0,0,:],'rx')
# plot the ground truth
plt.plot(t[0,0,0,:],y_true[0,0,0,:],'r')
# plot the fit using the inital guess (for reference)
mn = vae_norm_init.sess.run(vae_norm_init.mp_mean)
y_est = sess.run(nlinmod.evaluate(mn[:,:,:,0:nlinmod.nparams],t))
plt.plot(t[0,0,0,:],y_est[0,0,0,:],'k.')
# plto the fit with the estimated parameter values
mn = vae_norm.sess.run(vae_norm.mp_mean)
y_est = sess.run(nlinmod.evaluate(mn[:,:,:,0:nlinmod.nparams],t))
plt.plot(t[0,0,0,:],y_est[0,0,0,:],'b')