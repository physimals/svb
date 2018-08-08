# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:13:44 2016

@author: woolrich
"""

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
                 learning_rate=0.001, batch_size=100, mode_corr='infer_post_corr', do_folded_normal=0, vae_init=None, mp_mean_init=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mode_corr=mode_corr
        self.n_modelparams=2
        self.do_folded_normal=do_folded_normal
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, 1])
        
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
                        
        # generative model p(x | mu) = N(mu, varmu)
        # prior p(mp)~MVN(mp_mean_0,mp_covar_0)
        # where mp=model_params
        # e.g.  mp[0] is mu (but is log(mu) if doing folded normal)
        #       mp[1] is log(varmu)
        # approximating distributions: 
        # q(mp) ~ MVN(mp_mean, mp_covar)
                                        
        # -- Prior:         
        
        # Define priors for the parameters (i.e. the mean and var of the distirbution we are trying to infer) - these are TF constants for the mean and covariance matrix of an MVN (as above)
        self.mp_mean_0 = tf.zeros([self.n_modelparams], dtype=tf.float32)         
        self.mp_covar_0 = tf.diag(tf.constant(100.0, shape=[self.n_modelparams], dtype=tf.float32))
        
        # -- Approximating Posterior:
        
        # Setup and initialise mp_mean 
        # This is the posterior means for the parameters in the approximating distribution
        if vae_init==None:
            if mp_mean_init[0][0]==None:
                mp_mean_init=0.1*tf.random_normal([1,self.n_modelparams],0,1, dtype=tf.float32)
            self.mp_mean = tf.Variable(mp_mean_init, dtype=tf.float32, name='mp_mean')
        else:
            self.mp_mean = tf.Variable(vae_init.sess.run(vae_init.mp_mean), dtype=tf.float32, name='mp_mean')
                
        # Setup mp_covar
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
        mu=tf.reshape(self.mp[0,:],[self.batch_size,1])
        log_var_mu=tf.reshape(self.mp[1,:],[self.batch_size,1]) #remebering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)
        
        if self.do_folded_normal:
            # use a iid folded normal distribution
        
            # mean is in log space to enforce positivity
            mu = tf.exp(mu)
            
            reconstr_loss = 0.5 * tf.div(tf.add(tf.square(self.x), tf.square(mu)), tf.exp(log_var_mu)) + 0.5 * log_var_mu 
            reconstr_loss = reconstr_loss - self._log_cosh(tf.div(tf.multiply(self.x,mu), tf.exp(log_var_mu)))
            reconstr_loss = tf.reduce_sum(reconstr_loss,0)
        else:
            
            # use a iid normal distribution
            # Calculate the loglikelihood given our supplied mp values
            reconstr_loss = tf.reduce_sum(0.5 * tf.div(tf.square(tf.subtract(self.x,mu)), tf.exp(log_var_mu)) + 0.5 * log_var_mu ,0)
                                          
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
        
    def partial_fit(self, X):
        """Train model based on mini-batch of input data.       
        Return cost of mini-batch.
        """
        
        # Do the optimization (self.optimizer), but also calcuate the cost for reference (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been provided X         
        opt, cost = self.sess.run((self.optimizer, self.cost), feed_dict={self.x: X})
        return cost

####
#%%

def train(data, mode_corr='infer_post_corr', learning_rate=0.01,
          batch_size=100, training_epochs=10, display_step=1, do_folded_normal=False, vae_init=None, mp_mean_init=None):
    vae = VaeNormalFit(mode_corr=mode_corr, learning_rate=learning_rate, 
                                 batch_size=batch_size, do_folded_normal=do_folded_normal, vae_init=vae_init, mp_mean_init=mp_mean_init)
    
    cost_history=np.zeros([training_epochs])        
    
    # Training cycle
    for epoch in range(training_epochs):

        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        
        #print("total_batch:", '%d' % total_batch )
        
        index = 0
        
        batch_xs = data[index:index+batch_size]
        # print(vae.sess.run((vae.log_diag_chol_mp_covar,vae.mp_covar), feed_dict={vae.x: batch_xs}))
         
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = data[index:index+batch_size]
            index = index + batch_size
                     
            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
                        
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
#%%

class twodGridInfer(object):
    def __init__(self, data, gridsize=[20, 21], ranges=[-10, 10, 0.1, 10], do_folded_normal=0):
        self.n_modelparams=2 # This is a 2D grid search over a range of plausible mean and variance values
        self.gridsize=gridsize
        self.ranges=ranges
        self.data=data
        self.do_folded_normal=do_folded_normal
        
        self._setup_grid()        
        
        self.cost = self._cost_function() #calcuate the cost funciton, which is the log posterior at each gid location
        self.posterior = np.exp(-self.cost) # convert to posterior (rather than log posterior)
        
    def _cost_function(self):  
        cost=0
        for tt in range(len(self.data)): #consider each sample in turn (i.e. each data point)
            xvals=self.x_values[self.gridpoints[0]] #these are the possible mean values (in our grid search)
            yvals=self.y_values[self.gridpoints[1]] #these are the possible variance values
            datatt_tiled=np.tile(self.data[tt],xvals.shape)
        
            if self.do_folded_normal:
                # use a iid folded normal distribution
                cost = cost + 0.5*np.divide(np.square(datatt_tiled)+np.square(xvals),yvals) + 0.5 * np.log(yvals)
                cost = cost - np.log(1e-5+np.cosh(np.divide(np.multiply(datatt_tiled,xvals),yvals)))
                #import pdb; pdb.set_trace()
                
            else:
                # use a iid normal distribution
                # calculate the log posterior (well likelihood as there is no prior defined here)
                cost = cost + 0.5*np.divide(np.square(datatt_tiled-xvals),yvals) + 0.5 * np.log(yvals)
                # the cost function is the log posterior summed over all data points (it would of course be the product for non log)
            
        return np.transpose(cost)
        
    def _setup_grid(self): 
        self.x_values=np.linspace(self.ranges[0], self.ranges[1], self.gridsize[0])
        self.y_values=np.linspace(self.ranges[2], self.ranges[3], self.gridsize[1])
                    
        self.gridpoints=np.mgrid[0:self.gridsize[0],0:self.gridsize[1]]   


def folded_norm_pdf(x, mean, variance):    
    meantile=np.tile(mean,x.shape)
    vartile=np.tile(variance,x.shape)
    pdf = np.exp(-np.divide(np.square(x)+np.square(meantile),2*vartile))
    pdf = np.multiply(pdf, np.cosh(np.divide(np.multiply(x,meantile),vartile)))
    pdf = np.multiply(pdf, np.sqrt(2/(np.pi*vartile)))    
    return pdf

################################################################################################
################################################################################################   
#%% main code

#np.random.seed(0)
#tf.set_random_seed(0)
plt.close("all")

############
#%% parse args

parser = argparse.ArgumentParser(description='Simulate and then infer, a normal or folded normal distribution')
parser.add_argument("--n_samples", type=int, default=100, help="number of samples", required=False)
parser.add_argument("--true_mean", type=float, default=1, help="true mean of simulated data", required=False)
parser.add_argument("--true_var", type=float, default=4, help="true var of simulated data", required=False)
parser.add_argument("--sim_folded_normal", action='store_true', help="simulate folded normal?", required=False)
parser.add_argument("--infer_folded_normal", action='store_true', help="infer folded normal?", required=False)
args = parser.parse_args()

n_samples = args.n_samples #An integer
true_mean = [args.true_mean] #We need a 1D array - create from the float value
true_var = [[float(args.true_var)]] #We need a 2D array - create from the float value
#import pdb; pdb.set_trace()

sim_folded_normal = args.sim_folded_normal
infer_folded_normal = args.infer_folded_normal
print(sim_folded_normal)

############
#%% create simulated data

#
x = np.random.multivariate_normal(true_mean, true_var, n_samples).astype('float32')

if sim_folded_normal:
    x=np.abs(x)
    plotfrom=-np.sqrt(true_var)
    plotto=6*np.sqrt(true_var)
else:
    plotfrom=true_mean-3*np.sqrt(true_var)
    plotto=true_mean+3*np.sqrt(true_var)

#def log_mvn_pdf(theta, mu, covar):
#    mn=tf.sub(theta,mu)
#    return -0.5*tf.log(tf.det(covar))-0.5*tf.matmul(tf.matmul(mn,tf.inv(covar)),tf.transpose(mn))  

############
#%% do grid approach

if infer_folded_normal:
    ranges=[0, 5, 0.01, 15]    
else:  
    ranges=[-5, 5, 0.1, 8]
    
twod=twodGridInfer(x, gridsize=[200, 201], do_folded_normal=infer_folded_normal, ranges=ranges)
# at this point twod.posterior is a 200x201 array of posterior values

do_plot=True
if do_plot:
    plt.figure(2)
    imgplot = plt.imshow(np.flipud(twod.posterior), extent=twod.ranges)
    plt.xlabel('mean')
    plt.ylabel('variance')
  
############ 
#%% do vae approach  
 
infer_folded_normal=sim_folded_normal

learning_rate=0.02
batch_size=n_samples
training_epochs=400

# initialise params
mp_mean_init=np.zeros([1,2]).astype('float32')
# set mean to 2x the sample mean to give vae something to do!
init_mean=2*np.mean(x)
init_var=np.var(x)
print("VAE Init mean:", "{:.9f}".format(init_mean), "Init var=", "{:.9f}".format(init_var))

if infer_folded_normal:
    mp_mean_init[0,0]=np.log(init_mean)
else:
    mp_mean_init[0,0]=init_mean
 
mp_mean_init[0,1]=np.log(init_var)
  
# call with no training epochs to get initialisation
mode_corr='infer_post_corr'            
vae_norm_init, cost_history = train(x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=0, batch_size=batch_size, do_folded_normal=infer_folded_normal, mp_mean_init=mp_mean_init)

# now train with correlation between mean and variance
mode_corr='no_post_corr'
vae_norm_no_post_corr, cost_history_no_post_corr = train(x, mode_corr=mode_corr, learning_rate=learning_rate, training_epochs=training_epochs, batch_size=batch_size, do_folded_normal=infer_folded_normal, vae_init=vae_norm_init)

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
