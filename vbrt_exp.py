# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:13:44 2016

@author: chappell
"""

# derived from vbrt_tute - for a simple (epxonential) non-linear forward model

#%%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse

############
#%% VAE obj

class VaeNormalFit(object):
    def __init__(self,  
                 learning_rate=0.001, batch_size=100, mode_corr='infer_post_corr', do_folded_normal=0, vae_init=None, mp_mean_init=None, scale=1):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mode_corr=mode_corr
        self.n_modelparams=3
        self.scale=scale
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, 1])
        
        # tf fixed model variables (in this case it is the time values)
        self.t = tf.placeholder(tf.float32, [None, 1])
        
        # Create model
        self._create_model(vae_init, mp_mean_init)
        
        # Define loss function based variational upper-bound and 
        # corresponding optimizer
        self._create_loss_optimizer()
        
        # Initializing the tensor flow variables
        init = tf.initialize_all_variables()

        # Launch the session
        self.sess = tf.Session()
        
        self.sess.run(init)
        
    def _create_model(self, vae_init, mp_mean_init):
                        
        # generative model p(x | mu) = N(A*exp(-t*R1), var)
        # prior p(mp)~MVN(mp_mean_0,mp_covar_0)
        # where mp=model_params
        # e.g.  mp[0] is A
        #       mp[1] is R1
        #       mp[2] is log(var)
        # approximating distributions: 
        # q(mp) ~ MVN(mp_mean, mp_covar)
                                        
        # -- Prior:         
        
        # Define priors for the parameters  - these are TF constants for the mean and covariance matrix of an MVN (as above)
        self.mp_mean_0 = tf.zeros([self.n_modelparams], dtype=tf.float32)         
        self.mp_covar_0 = tf.diag(tf.constant(100.0, shape=[self.n_modelparams], dtype=tf.float32))
        
        # -- Approximating Posterior:
        
        # Setup and initialise mp_mean (as a Variable to be optimised later)
        # This is the posterior means for the parameters in the approximating distribution
        if vae_init==None:
            if mp_mean_init[0][0]==None:
                mp_mean_init=0.1*tf.random_normal([1,self.n_modelparams],0,1, dtype=tf.float32)
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
                    self.off_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((nn,nn),0,0.1,dtype=tf.float32), name='off_diag_chol_mp_covar')                    
                else:
                    self.off_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.off_diag_chol_mp_covar), name='off_diag_chol_mp_covar')                    
                
                #import pdb; pdb.set_trace()
                       
                # diag: this needs to setup separately to the off-diag as it needs to be positive
                if vae_init==None:
                    self.log_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((self.n_modelparams,),-2,0.4, dtype=tf.float32), name='log_diag_chol_mp_covar')                    
                else:
                    self.log_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.log_diag_chol_mp_covar), name='log_diag_chol_mp_covar')                    

                diag_chol_mp_covar_mat = tf.diag(tf.exp(self.log_diag_chol_mp_covar), name='chol_mp_covar')

                # reshape to matrix
                self.chol_mp_covar = tf.add((diag_chol_mp_covar_mat),tf.matrix_band_part(self.off_diag_chol_mp_covar, -1, 0))

            else:
                if vae_init==None:
                    self.chol_mp_covar = tf.Variable(0.05+tf.diag(tf.constant(0.1,shape=(self.n_modelparams,))), dtype=tf.float32, name='chol_mp_covar')                            
                else:
                    self.chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.chol_mp_covar), name='chol_mp_covar')                    
        
        else:     
            # infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            if vae_init==None:
                self.log_diag_chol_mp_covar = tf.Variable(tf.constant(-2,shape=(self.n_modelparams,), dtype=tf.float32), name='log_diag_chol_mp_covar')                    
            else:
                self.log_diag_chol_mp_covar = tf.Variable(vae_init.sess.run(vae_init.log_diag_chol_mp_covar), name='log_diag_chol_mp_covar')                    
            
            self.chol_mp_covar = tf.diag(tf.exp(self.log_diag_chol_mp_covar), name='chol_mp_covar')    
        
        # form the covariance matrix from the chol decomposition
        self.mp_covar = tf.matmul(tf.transpose(self.chol_mp_covar),self.chol_mp_covar, name='mp_covar')
        
        # -- Generate sample of mp:
        # Use reparameterisation trick                
        # mp = mp_mean + chol(mp_covar)*eps
        
        eps = tf.random_normal((self.n_modelparams, self.batch_size), 0, 1, dtype=tf.float32)
        self.eps=eps
               
        self.mp = tf.add(tf.tile(tf.reshape(self.mp_mean,[self.n_modelparams,1]),[1,self.batch_size]), tf.matmul(self.chol_mp_covar, eps))
                        
    
    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
    
        # 1.) log likelihood
           
        #import pdb; pdb.set_trace()
           
        # unpack params    
        A = tf.reshape(self.mp[0,:],[self.batch_size,1])
        R1 = tf.reshape(self.mp[1,:],[self.batch_size,1])
        log_var = tf.reshape(self.mp[2,:],[self.batch_size,1]) #remebering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)
        
        # use a iid normal distribution
        # Calculate the loglikelihood given our supplied mp values
        y_pred = A * tf.exp(-R1 * self.t)
        reconstr_loss = tf.reduce_sum(0.5 * self.scale * ( tf.div(tf.square(tf.subtract(self.x,y_pred)), tf.exp(log_var)) + log_var + np.log( 2 * np.pi) ) ,0)
        # NOTE: scale (the relative scale of number of samples and size of batch) appears in here to get the relative scaling of log-likehood correct, even though we have dropped the constant term log(n_samples)                              
            
        self.reconstr_loss=reconstr_loss
        
        # 2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        # latent_loss = log_mvn_pdf(self.mp, self.mp_mean, mp_covar) \
        #                    - log_mvn_pdf(self.mp, self.mp_mean_0, self.mp_covar_0)
       
        # 2.) D_KL{q(mp)~ MVN(mp_mean, mp_covar), p(mp)~ MVN(mp_mean_0, mp_covar_0)}
        
        #import pdb; pdb.set_trace()
                 
        inv_mp_covar_0=tf.matrix_inverse(self.mp_covar_0)
        mn=tf.reshape(tf.subtract(self.mp_mean,self.mp_mean_0),[self.n_modelparams,1])
        
        latent_loss = 0.5*(tf.trace(tf.matmul(inv_mp_covar_0,self.mp_covar))+tf.matmul(tf.matmul(tf.transpose(mn),inv_mp_covar_0),mn) - self.n_modelparams + tf.log(tf.matrix_determinant(self.mp_covar_0, name='det_mp_covar_0') ) - tf.log(tf.matrix_determinant(self.mp_covar, name='det_mp_covar') ) )
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

def train(t,data, mode_corr='infer_post_corr', learning_rate=0.01,
          batch_size=100, training_epochs=10, display_step=1, do_folded_normal=False, vae_init=None, mp_mean_init=None):
    
    # need to determine the 'scale' between log-likihood and KL(post-prior) when using batches
    scale = n_samples/batch_size
    
    vae = VaeNormalFit(mode_corr=mode_corr, learning_rate=learning_rate, 
                                 batch_size=batch_size, scale=scale, do_folded_normal=do_folded_normal, vae_init=vae_init, mp_mean_init=mp_mean_init)
    
    cost_history=np.zeros([training_epochs]) 
    total_batch = int(n_samples / batch_size)
    #print("total_batch:", '%d' % total_batch )
    #cost_full_history =  np.zeros([training_epochs * total_batch])      
    
    # Training cycle
    for epoch in range(training_epochs): # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.

        avg_cost = 0.
        index = 0
        
        batch_xs = data[index:index+batch_size]
        t_xs = t[index:index+batch_size]
        # print(vae.sess.run((vae.log_diag_chol_mp_covar,vae.mp_covar), feed_dict={vae.x: batch_xs}))
         
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data[index:index+batch_size]
            t_xs = t[index:index+batch_size]
            index = index + batch_size
                     
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
n_samples=100
true_amp = 1.0 # the true amplitude of the model
true_R1 = 1.0 # the true decay rate
true_sd = 0.1 # the s.d. of the noise
true_var = true_sd*true_sd

#time points
t_end = 5*1/true_R1 #for time being, make sure we have a good range of samples based on the choice of R1
t = np.linspace(0,t_end,num=n_samples)
t = np.reshape(t,(-1,1))

############
#%% create simulated data

# calcuate forward model
y_true = true_amp * np.exp(-t*true_R1)

# add noise
x = np.random.normal(y_true,true_sd)
x = np.reshape(x, (-1,1))

#%% do vae approach  


learning_rate=0.02
batch_size=10 #n_samples 
#scale = n_samples/batch_size
training_epochs=400

# initialise params
mp_mean_init=np.zeros([1,3]).astype('float32')
# some roughtloy sensible init values from the data
init_amp = np.max(x)
init_R1 = 1.0
init_var=np.var(x)
mp_mean_init[0,0]=init_amp
mp_mean_init[0,1]=init_R1
mp_mean_init[0,1]=np.log(init_var) #NB becuase we happen to be infering the log of the variance
  
# call with no training epochs to get initialisation
mode_corr='infer_post_corr'            
vae_norm_init, cost_history = train(t, x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=0, batch_size=batch_size, mp_mean_init=mp_mean_init)

# now train with no correlation between mean and variance
mode_corr='no_post_corr'
vae_norm_no_post_corr, cost_history_no_post_corr = train(t, x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, vae_init=vae_norm_init)

#%%
# now train with correlation between mean and variance
mode_corr='infer_post_corr'
vae_norm, cost_history = train(x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, do_folded_normal=infer_folded_normal, vae_init=vae_norm_init)

#mn = vae_norm.sess.run(vae_norm.mp_mean)
#import pdb; pdb.set_trace()
#print("VAE Estimated mean:", "{:.9f}".format(mn[0,0]), "Estimated var=", "{:.9f}".format(np.exp(mn[0,1])))    
#print("True mean:", "{:.9f}".format(true_mean[0]), "True var=", "{:.9f}".format(true_var[0][0]))

# plot cost history
plt.figure(3)

ax1 = plt.subplot(1,2,1)

plt.subplots_adjust(hspace=2)
plt.subplots_adjust(wspace=0.8)

plt.plot(cost_history_no_post_corr)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title('No post correlation')

ax2 = plt.subplot(1,2,2)
plt.plot(cost_history)
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title('Infer post correlation')

# compute images for plotting the vae MVN approximate posterior on the same grid as the 2Dgrid posterior
xvals=twod.x_values[twod.gridpoints[0]]
xvals.shape = (xvals.shape[0], xvals.shape[1], 1)
if infer_folded_normal:
    xvals = np.log(xvals)
        
yvals=np.log(twod.y_values[twod.gridpoints[1]])
yvals.shape = (yvals.shape[0], yvals.shape[1], 1)
rvs=np.concatenate((xvals,yvals),2)

mn = vae_norm_init.sess.run(vae_norm_init.mp_mean)
vae_post_init=stats.multivariate_normal.pdf(rvs, mean=mn[0], cov=vae_norm_init.sess.run(vae_norm_init.mp_covar))
mn = vae_norm_no_post_corr.sess.run(vae_norm_no_post_corr.mp_mean)
vae_post_no_post_corr=stats.multivariate_normal.pdf(rvs, mean=mn[0], cov=vae_norm_no_post_corr.sess.run(vae_norm_no_post_corr.mp_covar))
mn = vae_norm.sess.run(vae_norm.mp_mean)
vae_post=stats.multivariate_normal.pdf(rvs, mean=mn[0], cov=vae_norm.sess.run(vae_norm.mp_covar))

# do the posterior plots
plt.figure(4)

plt.subplots_adjust(hspace=0.5, wspace=0.5)

ax1 = plt.subplot(2,2,1)

# plot the histogram of the sampled data
hist, bins = np.histogram(x, bins=50, normed=True)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.xlabel('data')
plt.grid(True)
if sim_folded_normal:
    pd=np.transpose(folded_norm_pdf(bins,true_mean[0],true_var[0][0]))   
else:
    pd=np.transpose(stats.norm.pdf(bins,true_mean,np.sqrt(true_var)))
plt.plot(bins, pd, 'r-', lw=5, alpha=0.6, label='norm pdf')

ax1 = plt.subplot(2,2,2)
imgplot = plt.imshow(np.flipud(twod.posterior), extent=twod.ranges, aspect='auto')
plt.xlabel('mean')
plt.ylabel('variance')
plt.title('Exhastive Grid Posterior')

if False:
    imgplot = plt.imshow(np.flipud(np.transpose(vae_post_init)), extent=twod.ranges, aspect='auto')
    plt.xlabel('mean')
    plt.ylabel('variance')
    plt.title('VBRT Posterior Init')

ax1 = plt.subplot(2,2,3)
imgplot = plt.imshow(np.flipud(np.transpose(vae_post_no_post_corr)), extent=twod.ranges, aspect='auto')
plt.xlabel('mean')
plt.ylabel('variance')
plt.title('VBRT No Post. Corr')

ax1 = plt.subplot(2,2,4)
imgplot = plt.imshow(np.flipud(np.transpose(vae_post)), extent=twod.ranges, aspect='auto')
plt.xlabel('mean')
plt.ylabel('variance')
plt.title('VBRT With Post. Corr')

plt.show()

#import pdb; pdb.set_trace()

# source activate tensorflow14                        
# python /Users/woolrich/Dropbox/vols_scripts/vbrt_tute/vbrt_tute.py --true_mean=1 --true_var=8 --n_samples=50 --sim_folded_normal --infer_folded_normal
# python /Users/woolrich/Dropbox/vols_scripts/vbrt_tute/vbrt_tute.py --true_mean=1 --true_var=4 --n_samples=30 
