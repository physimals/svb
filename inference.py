"""
Stochastic Bayesian inference of a nonlinear model
"""
import numpy as np
import tensorflow as tf

class VaeNormalFit(object):
    """
    Stochastic Bayesian model fitting

    Infers:
     - Mean values of model parameters
     - log(noise variance)
     - A covariance matrix in which the diagonal elements are log variances
    """
    def __init__(self, model, vae_init=None, **kwargs):
        self.model = model

        # Learning rate for the optimizer
        self.learning_rate = kwargs.get("learning_rate", 0.02)

        # Each learning epoch processes a batch of data
        self.batch_size = kwargs.get("batch_size", 100)

        # Number of samples per parameter for the sampling of the prior distribution
        self.draw_size = kwargs.get("draw_size", 100)

        # Inference mode for the posterior variance - if set to INFER_POST_CORR then
        # co-variances will be estimated, otherwise only parameter variances (diagonal
        # elements) will be estimated
        self.mode_corr = kwargs.get("mode_corr", "infer_post_corr")
        self.do_folded_normal = kwargs.get("do_folded_normal", 0)
        
        # The total number of parameters to infer - model parameters plus noise parameters
        self.nparams = model.nparams + 1
        
        # Set up the tensorflow graph which will be trained to do the inference
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)
            
            # Tensorflow parameters required for the training run - x and t are
            # the data and time values of the batch being trained against, xfull
            # is the full data (used to initialize the posterior in some cases)
            self.x = tf.placeholder(tf.float32, [None])
            self.xfull = tf.placeholder(tf.float32, [None])
            self.t = tf.placeholder(tf.float32, [None])
            
            # Create prior distribution, initial posterior and samples from the posterior
            self._init_prior()
            self._init_posterior(vae_init)
            self._create_samples()

            # Define loss function based variational upper-bound and corresponding optimizer
            self._create_loss_optimizer()

            # Variable initializer
            self.init = tf.global_variables_initializer()

            # Tensorflow session for runnning graph
            self.sess = tf.Session()
        
    def _init_prior(self):
        """
        Define prior distribution
        
        These are TF constants for the mean and covariance matrix of an MVN. The
        prior covariance matrix is always diagonal (no inter-parameter correlations)
        """
        prior_means = np.zeros(self.nparams)
        prior_vars = 100*np.ones(self.nparams)
        for idx, param in enumerate(self.model.params):
            prior_means[idx] = param.prior_mean
            prior_vars[idx] = param.prior_var
            
        self.mp_mean_0 = tf.constant(prior_means, dtype=tf.float32, name="mp_mean_0")  
        self.mp_covar_0 = tf.diag(tf.constant(prior_vars, dtype=tf.float32), name="mp_covar_0")
        
    def _init_posterior(self, vae_init):
        """
        Define initial approximating posterior distribution

        If a previous run is given it is used to initialize the posterior, otherwise
        the model is given the opportunity to set the initial posterior based on the
        data.

        The posterior is always determined in terms of the Cholesy decomposition - this
        ensures the covariance matrix is positive definite. This means there are always
        two tensors defined in the graph ``log_diag_chol_mp_covar`` which is the log of
        the diagonal of the matrix and ``off_diag_chol_mp_covar`` which are the off-diagnoal
        elements. 
        
        If ``mode_corr`` is ``NO_POST_CORR`` then the latter is a constant
        containing zeros which plays no role in the inference - it exists only so that
        a subsequent run with correlation enabled can be performed using the non-correlated
        run as initialization.
        """
        if vae_init is not None:
            # Set self.mp_mean and self.chol_mp_covar from a previous run
            self._init_posterior_chol_from_vae(vae_init)
        else:
            # Set self.mp_mean and self.chol_mp_covar from the model / data
            self._init_posterior_chol_from_model()
            
        # form the covariance matrix from the chol decomposition
        self.mp_covar = tf.matmul(tf.transpose(self.chol_mp_covar), self.chol_mp_covar, name='mp_covar')
    
        if self.mode_corr != 'infer_post_corr':
            tf.constant(np.zeros([self.nparams, self.nparams], dtype=np.float32), name='off_diag_chol_mp_covar')
        
        # For preventing singular covariance matrix
        self.reg_cov = 1e-6 * tf.constant(np.identity(self.nparams, dtype=np.float32), name="reg_cov")

    def _init_posterior_chol_from_model(self):
        """
        Initialize posterior mean and Cholesky decomposition of posterior covariance from
        the model and the full data
        """
        # Initialize posterior means for the parameters in the approximating distribution
        post_means = list(0.1 * np.random.normal(0, 1, size=self.nparams))
        self.model.update_initial_posterior(self.t, self.xfull, post_means)

        # Initialize noise posterior mean using data variance - NB we happen to be infering the log of the variance
        mean, variance = tf.nn.moments(self.xfull, axes=[0])
        post_means[self.nparams - 1] = tf.log(variance)

        self.mp_mean = tf.Variable(tf.stack(post_means), dtype=tf.float32, name="mp_mean")
                
        # Setup post_covar (as a Variable to be optimised later)
        # This is the posterior covariance matrix in the approximating distribution
        # Note that parameterising using chol ensures that mp_covar is positive def, although note that this does not reduce the number of params accordingly (need a tf.tril() func)    
        if self.mode_corr == 'infer_post_corr':
            # Infer a full covariance matrix with on and off-diagonal elements     
            if True:
                off_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((self.nparams,self.nparams),0,0.1,dtype=tf.float32, seed=1), name='off_diag_chol_mp_covar')
                log_diag_chol_mp_covar = tf.Variable(tf.truncated_normal((self.nparams,),-2,0.4, dtype=tf.float32, seed=1), name='log_diag_chol_mp_covar')
            
                # Combine diagonal and off-diagonal elements and reshape to matrix
                diag_chol_mp_covar_mat = tf.diag(tf.exp(log_diag_chol_mp_covar))
                self.chol_mp_covar = tf.add((diag_chol_mp_covar_mat), tf.matrix_band_part(off_diag_chol_mp_covar, -1, 0), name='chol_mp_covar')

            else:
                # Alternative not inferring the log of the diagnoal
                self.chol_mp_covar = tf.Variable(0.05+tf.diag(tf.constant(0.1,shape=(self.nparams,))), dtype=tf.float32, name='chol_mp_covar')          
        
        else:     
            # Infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            log_diag_chol_mp_covar = tf.Variable(tf.constant(-2,shape=(self.nparams,), dtype=tf.float32), name='log_diag_chol_mp_covar')               
            
            self.chol_mp_covar = tf.diag(tf.exp(log_diag_chol_mp_covar), name='chol_mp_covar')    

    def _init_posterior_chol_from_vae(self, vae_init):
        """
        Initialize posterior mean and Cholesky decomposition of posterior covariance from
        the model and the full data
        """
        # The initial posterior mean is taken directly from the initializing distribution
        self.mp_mean = tf.Variable(vae_init.mp_mean_out, name="mp_mean") 

        # If we are inferring the off diagonal elements then initialize them from the previous run. Note
        # that they may be zero if the previous run did not infer them itself.
        log_diag_chol_mp_covar = tf.Variable(vae_init.output('log_diag_chol_mp_covar'), name='log_diag_chol_mp_covar')
        if self.mode_corr == 'infer_post_corr':
            off_diag_chol_mp_covar = tf.Variable(vae_init.output('off_diag_chol_mp_covar'), name='off_diag_chol_mp_covar')
            diag_chol_mp_covar_mat = tf.diag(tf.exp(log_diag_chol_mp_covar))
            self.chol_mp_covar = tf.add((diag_chol_mp_covar_mat), tf.matrix_band_part(off_diag_chol_mp_covar, -1, 0), name='chol_mp_covar')
        else:
            self.chol_mp_covar = tf.diag(tf.exp(log_diag_chol_mp_covar), name='chol_mp_covar')    
            
    def _create_samples(self):
        """
        Create samples of parameter values from the current posterior distribution

        This is done by initially forming samples from a (0, 1) Gaussian and then
        scaling by the posterior mean and std.dev. ``self.draw_size`` unique samples
        are created but are then tiled to form ``self.batch_size`` actual samples
        """
        eps = tf.random_normal((self.nparams, self.draw_size), 0, 1, dtype=tf.float32, seed=1)

        # This seems to assume that draw_size is a factor of batch_size? If so, we end
        # up with batch_size samples for each parameter but only draw_size unique samples
        # FIXME is floor division right here?
        ntile = np.floor_divide(self.batch_size,self.draw_size)
        eps = tf.tile(eps,[1,ntile])
               
        # NB self.chol_mp_covar is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev. 
        sample = tf.tile(tf.reshape(self.mp_mean,[self.nparams,1]),[1,self.batch_size])
        self.mp = tf.add(sample, tf.matmul(self.chol_mp_covar, eps))
                        
    def _create_loss_optimizer(self):
        """
        Create the loss optimizer which will minimise the cost function

        The loss is composed of two terms:
    
        1.) log likelihood
        2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        """

        # Unpack noise parameter
        # Remembering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)
        log_var = tf.reshape(self.mp[self.nparams-1,:],[self.batch_size]) 

        # use a iid normal distribution
        # Calculate the loglikelihood given our supplied mp values
        y_pred = self.model.evaluate(self.mp, self.t)

        # This prediction currently has a different set of model parameters for each data point (i.e. it is not amortized) - due to the sampling process above
        # NOTE: scale (the relative scale of number of samples and size of batch) appears in here to get the relative scaling of log-likehood correct, even though we have dropped the constant term log(n_samples)
        scale = tf.to_float(tf.floordiv(tf.size(self.xfull), self.batch_size))
        self.reconstr_loss = tf.reduce_sum(0.5 * scale * ( tf.div(tf.square(tf.subtract(self.x, y_pred)), tf.exp(log_var)) + log_var + np.log( 2 * np.pi) ), 0, name="reconstr_loss")
        #self.reconstr_loss = tf.Print(self.reconstr_loss, [self.reconstr_loss], "reconstr")

        # 2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        # latent_loss = log_mvn_pdf(self.mp, self.mp_mean, mp_covar) \
        #                    - log_mvn_pdf(self.mp, self.mp_mean_0, self.mp_covar_0)       
        # 2.) D_KL{q(mp)~ MVN(mp_mean, mp_covar), p(mp)~ MVN(mp_mean_0, mp_covar_0)}
        
        inv_mp_covar_0 = tf.matrix_inverse(self.mp_covar_0)
        mn = tf.reshape(tf.subtract(self.mp_mean,self.mp_mean_0),[self.nparams,1])

        t1 = tf.trace(tf.matmul(inv_mp_covar_0,self.mp_covar))
        #t1 = tf.Print(t1, [t1], "t1")

        t2 = tf.matmul(tf.transpose(mn), inv_mp_covar_0)
        #t2 = tf.Print(t2, [t2], "t2")

        t3 = tf.matmul(t2, mn)
        #t3 = tf.Print(t3, [t3], "t3")

        t4 = tf.log(tf.matrix_determinant(self.mp_covar_0, name='det_mp_covar_0')) 
        #t4 = tf.Print(t4, [t4], "t4")

        #mp_covar = tf.Print(self.mp_covar, [self.mp_covar], "mp_covar", summarize=100)
        t5 = tf.log(tf.matrix_determinant(self.mp_covar + self.reg_cov, name='det_mp_covar') )
        #t5 = tf.Print(t5, [t5], "t5")

        latent_loss = 0.5*(t1 + t3 - self.nparams + t4 - t5)
        self.latent_loss = tf.identity(latent_loss, name="latent_loss")
        #self.latent_loss = tf.Print(self.latent_loss, [self.latent_loss], "latent")

        # Add the two terms (and average over the batch)
        self.cost = tf.reduce_mean(self.reconstr_loss + self.latent_loss, name="cost")   
        
        # Use ADAM optimizer
        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimizer = opt.minimize(self.cost)
        
    def initialize(self, Xfull):
        """
        Initialize global variables - i.e. initial values of posterior which
        may depend on the full data
        """
        self.sess.run(self.init, feed_dict={self.xfull: Xfull})

    def partial_fit(self, T, X, Xfull):
        """
        Train model based on mini-batch of input data.       
        Return cost of mini-batch.
        """
        # Do the optimization (self.optimizer), but also calcuate the cost for reference (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been provided X
        opt, cost, self.mp_mean_out, self.mp_covar_out, self.chol_mp_covar_out = self.sess.run((self.optimizer, self.cost, self.mp_mean, self.mp_covar, self.chol_mp_covar), feed_dict={self.xfull : Xfull, self.x: X, self.t: T})
        return cost

    def output(self, name=None):
        self.mp_mean_out, self.mp_covar_out, self.chol_mp_covar_out = self.sess.run((self.mp_mean, self.mp_covar, self.chol_mp_covar))
        if name:
            return self.sess.run((self.sess.graph.get_tensor_by_name("%s:0" % name),))[0]

    def train(self, t, data, training_epochs=100, display_step=1):
        """
        Train the graph to infer the posterior distribution given timeseries data 
        """
        n_samples = data.size
        n_batches = np.floor_divide(n_samples, self.batch_size)
        
        cost_history=np.zeros([training_epochs]) 
        
        # Training cycle
        self.initialize(data)
        #if training_epochs > 0:
        #    writer = tf.summary.FileWriter("graphs/new", self.sess.graph)
        
        for epoch in range(training_epochs): # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.
            avg_cost = 0.
            index = 0
            
            # Loop over all batches
            for i in range(n_batches):
                if 0:
                    # Batches are defined by sequential data samples
                    batch_xs = data[index:index+self.batch_size]
                    t_xs = t[index:index+self.batch_size]
                    index = index + self.batch_size
                else:
                    # Batches are defined by constant strides through the data samples
                    batch_xs = data[i::n_batches]
                    t_xs = t[i::n_batches]
                        
                # Fit training using batch data
                #print("initial cost", self.sess.run((self.reconstr_loss, self.latent_loss), feed_dict={self.xfull : data, self.x: batch_xs, self.t: t_xs}))
                cost = self.partial_fit(t_xs,batch_xs, data)
                            
                # Compute average cost
                avg_cost += cost / n_samples * self.batch_size

                if np.isnan(cost):
                    import pdb; pdb.set_trace()
                                                
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), \
                    "cost=", "{:.9f}".format(avg_cost))
            cost_history[epoch]=avg_cost   
            
        self.output()
        print("Final covar\n", self.mp_covar_out)
        return self, cost_history
