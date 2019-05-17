"""
Stochastic Bayesian inference of a nonlinear model
"""
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

class VaeNormalFit(object):
    """
    Stochastic Bayesian model fitting

    Infers:
     - Posterior mean values of model parameters
     - A posterior covariance matrix

    The general order for tensor dimensions is:
     - Voxel indexing (V=number of voxels)
     - Parameter indexing (P=number of parameters)
     - Time point indexing (T=number of time points)
     
    This ordering is chosen to allow the use of TensorFlow batch matrix
    operations. However it is inconvenient for the model which would like
    to be able to index input by parameter. For this reason we transpose
    when calling the model's ``evaluate`` function.

    Named Tensors
    
     - ``mp_mean_0`` P containing prior mean values for each parameter
     - ``mp_covar_0`` PxP containing prior covariance matrix (always diagnoal)
     - ``mp_mean`` VxP containing posterior mean values at each voxel
     - ``mp_covar`` VxPxP containing posterior covariance matrix for each voxel
     - ``chol_mp_covar_log_diag`` VxP containing the log of each parameter's variance
     - ``chol_mp_covar_off_diag`` VxPxP containing the off diagonal elements of 
       the Cholesky decomposition of ``mp_covar``.
     - ``reconstr_loss`` V containing the reconstr loss function for each voxel
     - ``latent_loss`` V containing the latent loss function for each voxel
     - ``cost`` Single value cost function based on sum of ``latent_loss`` and 
       ``reconstr_loss``
    """
    def __init__(self, model, vae_init=None, **kwargs):
        # The model to use for inference
        self.model = model

        # Learning rate for the optimizer
        self.learning_rate = kwargs.get("learning_rate", 0.02)

        # Inference mode for the posterior variance - if set to INFER_POST_CORR then
        # co-variances will be estimated, otherwise only parameter variances (diagonal
        # elements) will be estimated
        self.mode_corr = kwargs.get("mode_corr", "infer_post_corr")
        
        # The total number of parameters to infer - model parameters plus noise parameters
        self.nparams = model.nparams + 1

        # Debug mode
        self.debug = kwargs.get("debug", False)

        # Set up the tensorflow graph which will be trained to do the inference
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1)

            # Tensorflow input required for training
            # 
            # x will have shape VxB where B is the batch size and V the number of voxels
            # xfull is the full data so will have shape VxN where N is the full time size
            # t will have shape 1xB or VxB depending on whether the timeseries is voxel
            # dependent (e.g. in 2D multi-slice readout)
            # 
            # NB we don't know V and N at this stage so we set placeholder variables
            # self.nvoxels and self.nt and use validate_shape=False when creating 
            # tensorflow Variables
            self.x = tf.placeholder(tf.float32, [None, None])
            self.xfull = tf.placeholder(tf.float32, [None, None])
            self.t = tf.placeholder(tf.float32, [None, None])
            self.actual_learning_rate = tf.placeholder(tf.float32, shape=[])
         
            if self.debug:
                self.xfull = tf.Print(self.xfull, [self.xfull], "\nxfull", summarize=100)

            # Number of voxels in full data - known at runtime
            self.nvoxels = tf.shape(self.xfull)[0]

            # Number of time points in full data - known at runtime
            self.nt = tf.shape(self.xfull)[1]

            # Number of samples in each training batch - known at runtime
            self.batch_size = tf.shape(self.x)[1]
            
            # Number of samples per parameter for the sampling of the posterior distribution
            self.draw_size = kwargs.get("draw_size", self.batch_size)

            # Create prior distribution, initial posterior and samples from the posterior
            self._init_prior()
            self._init_posterior(vae_init)
            self._create_samples()

            # Define loss function based variational upper-bound and corresponding optimizer
            self._create_loss_optimizer()

            # Variable initializer
            self.init = tf.global_variables_initializer()

            # Variable saver
            self.saver = tf.train.Saver()

            # Tensorflow session for runnning graph
            self.sess = tf.Session()
            #if self.debug:
            #    self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        
    def _init_prior(self):
        """
        Define prior distribution
        
        These are TF constants for the mean and covariance matrix of an MVN. The
        prior covariance matrix is always diagonal (no inter-parameter correlations)
        """
        prior_means = np.zeros(self.nparams)
        prior_vars = np.zeros(self.nparams)
        for idx, param in enumerate(self.params):
            prior_means[idx], prior_vars[idx] = param.prior.nmean, param.prior.nvar
            
        # Noise prior mean is 0, variance 1e6 (i.e. non-informative)
        prior_vars[self.nparams-1] = 1e6

        self.mp_mean_0 = tf.constant(prior_means, dtype=tf.float32, name="mp_mean_0")  
        self.mp_covar_0 = tf.matrix_diag(tf.constant(prior_vars, dtype=tf.float32), name="mp_covar_0")
        if self.debug:
            self.mp_mean_0 = tf.Print(self.mp_mean_0, [self.mp_mean_0], "\nmp_mean_0", summarize=100)
            self.mp_covar_0 = tf.Print(self.mp_covar_0, [self.mp_covar_0], "\nmp_covar_0", summarize=100)

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
        mean_init, log_var_init = [None]*self.nparams, [None]*self.nparams
        covar_init = tf.truncated_normal((self.nvoxels, self.nparams, self.nparams), 0, 0.1, dtype=tf.float32)
        if vae_init is not None:
            mean_init = np.transpose(vae_init.output("mp_mean"))
            log_var_init = np.transpose(vae_init.output('log_diag_chol_mp_covar'))
            covar_init = vae_init.output('off_diag_chol_mp_covar')

        # Get initial posteriors for model parameters
        post_means = []
        post_log_vars = []
        for idx, param in enumerate(self.model.params):
            mean, log_var = param.initial(self.t, self.xfull, 
                                          mean_init=mean_init[idx],
                                          log_var_init=log_var_init[idx])
            post_means.append(mean)
            post_log_vars.append(log_var)

        # Noise FIXME need NoiseModel class for this
        mean, variance = tf.nn.moments(self.xfull, axes=[1])
        post_means.append(tf.Variable(tf.log(tf.maximum(variance, 1e-4)), validate_shape=False))
        post_log_vars.append(tf.Variable(tf.truncated_normal([self.nvoxels], -2, 0.4, dtype=tf.float32), validate_shape=False))

        # Stack initialized posterior mean and log-variance and transpose
        # so dimensions are (VxP) not (PxV)
        self.mp_mean = tf.transpose(tf.stack(post_means), name="mp_mean")
        log_diag_chol_mp_covar = tf.transpose(tf.stack(post_log_vars), name="log_diag_chol_mp_covar")
        
        # Create off-diagonal covariances to infer if required
        if self.mode_corr == 'infer_post_corr':
            # Infer a full covariance matrix with on and off-diagonal elements
            off_diag_chol_mp_covar = tf.Variable(covar_init, name='off_diag_chol_mp_covar', validate_shape=False)
            
            # Combine diagonal and off-diagonal elements into full matrix
            diag_chol_mp_covar_mat = tf.matrix_diag(tf.exp(log_diag_chol_mp_covar))
            self.chol_mp_covar = tf.add((diag_chol_mp_covar_mat), tf.matrix_band_part(off_diag_chol_mp_covar, -1, 0), name='chol_mp_covar')
        else:     
            # Define this constant in case we want to later use this to initialize a full correlation run
            tf.constant(np.zeros([self.nparams, self.nparams], dtype=np.float32), name='off_diag_chol_mp_covar')
        
            # Infer only diagonal elements of the covariance matrix - i.e. no correlation between the parameters
            self.chol_mp_covar = tf.matrix_diag(tf.exp(log_diag_chol_mp_covar), name='chol_mp_covar')    
        
        # Form the covariance matrix from the chol decomposition
        self.mp_covar = tf.matmul(tf.transpose(self.chol_mp_covar, perm=(0, 2, 1)), self.chol_mp_covar, name='mp_covar')
    
        # For preventing singular covariance matrix
        self.reg_cov = 1e-6 * tf.constant(np.identity(self.nparams, dtype=np.float32), name="reg_cov")

        # Define tensor containing output means in transformed model space
        model_means = []
        for idx, param in enumerate(self.model.params):
            model_means.append(param.dist.tomodel(self.mp_mean[:, idx]))
        tf.stack(model_means, name="mp_mean_model")
         
        if self.debug:
            self.chol_mp_covar = tf.Print(self.chol_mp_covar, [self.chol_mp_covar], "\nchol_mp_covar", summarize=100)
            self.mp_covar = tf.Print(self.mp_covar, [self.mp_covar], "\nmp_covar", summarize=100)
            self.mp_mean = tf.Print(self.mp_mean, [self.mp_mean], "\nmp_mean", summarize=100)
  
    def _create_samples(self):
        """
        Create samples of parameter values from the current posterior distribution

        This is done by initially forming samples from a (0, 1) Gaussian and then
        scaling by the posterior mean and std.dev. ``self.draw_size`` unique samples
        are created but are then tiled to form ``self.batch_size`` actual samples
        """
        eps = tf.random_normal((self.nvoxels, self.nparams, self.draw_size), 0, 1, dtype=tf.float32)

        # This seems to assume that draw_size is a factor of batch_size? If so, we end
        # up with batch_size samples for each parameter but only draw_size unique samples
        # FIXME is floor division right here?
        ntile = np.floor_divide(self.batch_size, self.draw_size)
        eps = tf.tile(eps, [1, 1, ntile])
               
        # NB self.chol_mp_covar is the Cholesky decomposition of the covariance matrix
        # so plays the role of the std.dev. 
        sample = tf.tile(tf.reshape(self.mp_mean, [self.nvoxels, self.nparams, 1]),[1, 1, self.batch_size])      
        if self.debug:
            sample = tf.Print(sample, [sample], "\nsample", summarize=100)

        self.mp = tf.add(sample, tf.matmul(self.chol_mp_covar, eps))              
        if self.debug:
            self.mp = tf.Print(self.mp, [self.mp], "\nmp", summarize=100)

    def _create_loss_optimizer_mvn(self):
        """
        Create the loss optimizer which will minimise the cost function

        The loss is composed of two terms:
    
        1. log likelihood. This is a measure of how likely the data are given the
           current posterior, i.e. how well the data fit the model using
           the inferred parameters.

        2. The latent loss. This is a measure of how closely the posterior fits the
           prior
        """

        ## Part 1: Log likelihood
        
        # Unpack noise parameter remembering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)
        noise_log_var = self.mp[:, -1, :]
        if self.debug:
            noise_log_var = tf.Print(noise_log_var, [noise_log_var], "\nnoise_log_var", summarize=100)

        # Transform the underlying Gaussian samples into the values required by the model
        # This depends on each model parameter's underlying distribution
        param_values = tf.transpose(self.mp, (1, 0, 2))
        model_values = []
        for idx, param in enumerate(self.model.params):
            model_values.append(param.dist.tomodel(self.mp[:, idx, :]))

        # Evaluate the model using the transformed values
        y_pred = self.model.evaluate(model_values, self.t)

        # Calculate the log likelihood given our supplied mp values
        # This prediction currently has a different set of model parameters for each data point (i.e. it is not amortized) - due to the sampling process above
        # NOTE: scale (the relative scale of number of samples and size of batch) appears in here to get the relative scaling of log-likehood correct, 
        # even though we have dropped the constant term log(n_samples)
        scale = tf.div(tf.to_float(self.nt), tf.to_float(self.batch_size))

        self.reconstr_loss = tf.reduce_sum(0.5 * scale * ( tf.div(tf.square(tf.subtract(self.x, y_pred)), tf.exp(noise_log_var)) + noise_log_var + np.log( 2 * np.pi) ), axis=1, name="reconstr_loss")
        if self.debug:
            self.reconstr_loss = tf.Print(self.reconstr_loss, [self.reconstr_loss], "\nreconstr", summarize=100)

        ## Part 1: Latent loss

        # 2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        # latent_loss = log_mvn_pdf(self.mp, self.mp_mean, mp_covar) \
        #                    - log_mvn_pdf(self.mp, self.mp_mean_0, self.mp_covar_0)       
        # 2.) D_KL{q(mp)~ MVN(mp_mean, mp_covar), p(mp)~ MVN(mp_mean_0, mp_covar_0)}
        
        inv_mp_covar_0 = tf.matrix_inverse(self.mp_covar_0)
        inv_mp_covar_0 = tf.tile(tf.reshape(inv_mp_covar_0, (1, self.nparams, self.nparams)), (self.nvoxels, 1, 1))
        mn = tf.subtract(self.mp_mean, tf.reshape(self.mp_mean_0, (1, -1)))

        t1 = tf.trace(tf.matmul(inv_mp_covar_0, self.mp_covar))
        t2 = tf.matmul(tf.reshape(mn, (self.nvoxels, 1, -1)), inv_mp_covar_0)
        t3 = tf.reshape(tf.matmul(t2, tf.reshape(mn, (self.nvoxels, -1, 1))), [self.nvoxels])
        t4 = tf.log(tf.matrix_determinant(self.mp_covar_0, name='det_mp_covar_0'))
        #t5 = tf.log(tf.matrix_determinant(self.mp_covar, name='det_mp_covar'))
        t5 = tf.log(tf.matrix_determinant(self.mp_covar + self.reg_cov, name='det_mp_covar'))

        if self.debug:
            t1 = tf.Print(t1, [t1], "t1")
            t3 = tf.Print(t3, [t3], "\nt3")
            t4 = tf.Print(t4, [t4], "t4")
            t5 = tf.Print(t5, [t5], "t5")

        latent_loss = 0.5*(t1 + t3 - self.nparams + t4 - t5)
        self.latent_loss = tf.identity(latent_loss, name="latent_loss")
        if self.debug:
            self.latent_loss = tf.Print(self.latent_loss, [self.latent_loss], "\nlatent", summarize=100)

        # Sum the cost from each voxel and use a single optimizer
        self.cost = tf.add(self.reconstr_loss, self.latent_loss, name="cost")
        self.mean_cost = tf.reduce_mean(self.cost, name="mean_cost")
        #self.cost = tf.square(tf.subtract(self.x, y_pred), name="cost")   
        #self.cost = tf.Print(self.cost, [self.cost], "cost", summarize=10)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.actual_learning_rate)
        self.optimize = self.optimizer.minimize(self.mean_cost)

    def _create_loss_optimizer(self):
        """
        Create the loss optimizer which will minimise the cost function

        The loss is composed of two terms:
    
        1. log likelihood. This is a measure of how likely the data are given the
           current posterior, i.e. how well the data fit the model using
           the inferred parameters.

        2. The latent loss. This is a measure of how closely the posterior fits the
           prior
        """

        ## Part 1: Log likelihood
        
        # Unpack noise parameter remembering that we are inferring the log of the variance of the generating distirbution (this was purely by choice)
        noise_log_var = self.mp[:, -1, :]
        if self.debug:
            noise_log_var = tf.Print(noise_log_var, [tf.shape(self.mp), tf.shape(self.mp_mean), tf.shape(self.mp_covar), tf.shape(self.mp_mean_0), tf.shape(self.mp_covar_0)], "shapes", summarize=100)

        # Transform the underlying Gaussian samples into the values required by the model
        # This depends on each model parameter's underlying distribution
        param_values = tf.transpose(self.mp, (1, 0, 2))
        model_values = []
        for idx, param in enumerate(self.model.params):
            model_values.append(param.dist.tomodel(self.mp[:, idx, :]))

        # Evaluate the model using the transformed values
        y_pred = self.model.evaluate(model_values, self.t)
        
        # Calculate the log likelihood given our supplied mp values
        # This prediction currently has a different set of model parameters for each data point (i.e. it is not amortized) - due to the sampling process above
        # NOTE: scale (the relative scale of number of samples and size of batch) appears in here to get the relative scaling of log-likehood correct, 
        # even though we have dropped the constant term log(n_samples)
        scale = tf.div(tf.to_float(self.nt), tf.to_float(self.batch_size))
        self.reconstr_loss = tf.reduce_sum(0.5 * scale * (tf.div(tf.square(tf.subtract(self.x, y_pred)), tf.exp(noise_log_var)) + noise_log_var), axis=1, name="reconstr_loss")
        if self.debug:
            self.reconstr_loss = tf.Print(self.reconstr_loss, [self.reconstr_loss], "\nreconstr", summarize=100)

        ## Part 1: Latent loss

        # 2.) log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        # latent_loss = log_mvn_pdf(self.mp, self.mp_mean, mp_covar) \
        #                    - log_mvn_pdf(self.mp, self.mp_mean_0, self.mp_covar_0)     
        
        # MVN pdf = (2pi)^(-k/2) |C|^(-1/2) exp(-0.5*(self.mp - self.mp_mean))
        
        mean_0 = tf.tile(tf.reshape(self.mp_mean_0, [1, self.nparams]), [self.nvoxels, 1])
        covar_0 = tf.tile(tf.reshape(self.mp_covar_0, [1, self.nparams, self.nparams]), [self.nvoxels, 1, 1])
        latent_loss = self.log_mvn_pdf(self.mp_mean, self.mp_covar, self.mp, "mp") - self.log_mvn_pdf(mean_0, covar_0, self.mp, "mp0")
        self.latent_loss = tf.identity(latent_loss, name="latent_loss")
        if self.debug:
            self.latent_loss = tf.Print(self.latent_loss, [self.latent_loss], "\nlatent", summarize=100)

        # Sum the cost from each voxel and use a single optimizer
        self.cost = tf.add(self.reconstr_loss, self.latent_loss, name="cost")
        self.mean_cost = tf.reduce_mean(self.cost, name="mean_cost")
        #self.cost = tf.square(tf.subtract(self.x, y_pred), name="cost")   
        #self.cost = tf.Print(self.cost, [self.cost], "cost", summarize=10)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.actual_learning_rate)
        self.optimize = self.optimizer.minimize(self.mean_cost)

    def log_mvn_pdf(self, mean, cov, values, name):
        """
        :param mean: [NV x P]
        :param cov: [NV x P x P]
        :param values: [NV x P x B]
        """
        #values = tf.Print(values, [values[95]], "\n%s_vals" % name, summarize=100)
        det_covar = tf.matrix_determinant(cov, name="%s_det" % name) # [NV]
        inv_covar = tf.matrix_inverse(cov, name="%s_inv" % name) # [NV, P, P]
        
        dx = tf.subtract(values, tf.expand_dims(mean, axis=-1)) # [NV x P x B]
        #dx = tf.Print(dx, [dx[95]], "\n%s_dxo" % name, summarize=100)

        dx = tf.expand_dims(tf.transpose(dx, [0, 2, 1]), axis=2, name="%s_dx" % name) # [NV x B x 1 x P]
        #dx = tf.Print(dx, [tf.shape(dx)], "dx", summarize=10)

        dxt = tf.reshape(dx, [self.nvoxels, self.batch_size, self.nparams, 1], name="%s_dxt" % name) # [NV x B x P x 1]
        #dxt = tf.Print(dxt, [dxt[95]], "\ndxt", summarize=100)

        inv_covar_tile = tf.tile(tf.reshape(inv_covar, [self.nvoxels, 1, self.nparams, self.nparams]), [1, self.batch_size, 1, 1]) # [NV x B x P x P]
        #inv_covar_tile = tf.Print(inv_covar_tile, [tf.shape(inv_covar_tile)], "inv_covar_tile", summarize=10)

        mul1 = tf.matmul(inv_covar_tile, dxt) # [NV x B x P x 1]
        mul2 = tf.matmul(dx, mul1, name="%s_mul" % name) # [NV x B x 1 x 1]
        #mul2 = tf.Print(mul2, [mul2[95]], "mul2", summarize=100)

        pdf = -0.5*(tf.tile(tf.reshape(tf.log(det_covar), [self.nvoxels, 1]), [1, self.batch_size]) + tf.reshape(mul2, [self.nvoxels, self.batch_size]))
        #pdf = tf.Print(pdf, [tf.reduce_mean(pdf, axis=1)[95]], "\n%s_pdf" % name, summarize=100)
        return tf.reduce_mean(pdf, axis=1)

    def initialize(self):
        """
        Initialize global variables - i.e. initial values of posterior which
        may depend on the full data
        """
        self.sess.run(self.init, feed_dict=self.feed_dict)
        # Save the initial posterior
        self.mp_mean_1 = self.output("mp_mean", feed_dict=self.feed_dict)
        self.mp_covar_1 = self.output("mp_covar", feed_dict=self.feed_dict)

    def fit_batch(self):
        """
        Train model based on mini-batch of input data.       

        :return: cost of mini-batch.
        """
        # Do the optimization (self.optimizer), but also calcuate the cost for reference (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been provided X
        _, cost = self.sess.run([self.optimize, self.cost], feed_dict=self.feed_dict)
        return cost

    def output(self, name, feed_dict=None):
        """
        Evaluate an output tensor

        e.g. ``output("mp_mean")`` returns the current posterior means
        """
        return self.sess.run((self.sess.graph.get_tensor_by_name("%s:0" % name),), feed_dict=self.feed_dict)[0]

    def train(self, t, data, batch_size, training_epochs=100, display_step=1, output_graph=None):
        """
        Train the graph to infer the posterior distribution given timeseries data 
        """
        # Expect t to have a dimension for voxelwise variation even if it is the same for all voxels
        if t.ndim == 1:
            t = t.reshape(1, -1)
        n_voxels = data.shape[0]
        n_samples = t.shape[1]
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        print(n_voxels, n_samples, n_batches)
        cost_history=np.zeros([training_epochs+1]) 
        param_history=np.zeros([training_epochs+1, self.model.nparams]) 
        
        # Training cycle
        self.feed_dict={
            self.xfull : data, 
            self.actual_learning_rate: self.learning_rate
        }
        self.initialize()
        if output_graph:
            writer = tf.summary.FileWriter(output_graph, self.sess.graph)
        
        trials, best_cost = 0, 1e12
        saved = False
        max_trials = 20
        for epoch in range(training_epochs): # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.
            err, avg_cost = 0, 0.0
            index = 0
            
            try:
                # Loop over all batches
                for i in range(n_batches):
                    if 0:
                        # Batches are defined by sequential data samples
                        batch_xs = data[:, index:index+batch_size]
                        t_xs = t[:, index:index+batch_size]
                        index = index + batch_size
                    else:
                        # Batches are defined by constant strides through the data samples
                        batch_xs = data[:, i::n_batches]
                        t_xs = t[:, i::n_batches]
                    
                    #batch_xs = np.tile(batch_xs, (1, 5))
                    #t_xs = np.tile(t_xs, (1, 5))
                    # Fit training using batch data
                    self.feed_dict[self.x] = batch_xs
                    self.feed_dict[self.t] = t_xs
                    cost = np.mean(self.fit_batch())
                                
                    # Compute average cost
                    avg_cost += cost / n_batches
            except tf.OpError:
                import traceback
                traceback.print_exc()
                err = 1

            mean_params = np.mean(self.output("mp_mean_model"), axis=1)
            latent = np.mean(self.output("latent_loss"))
            reconstr = np.mean(self.output("reconstr_loss"))
            if err or np.isnan(avg_cost) or np.any(np.isnan(mean_params)) or np.isnan(latent) or np.isnan(reconstr):
                self.feed_dict[self.actual_learning_rate] *= 0.9
                if saved:
                    self.saver.restore(self.sess, "/tmp/model.ckpt")
                sys.stdout.write("NaN values - reverting to previous best step with lower learning rate (%f)\n" % self.feed_dict[self.actual_learning_rate])
            else:
                # Display logs per epoch step
                cost_history[epoch] = avg_cost
                param_history[epoch, :] = mean_params
                if epoch % display_step == 0:
                    sys.stdout.write("Epoch: %04d, mean cost=%f (%f, %f), mean params=%s" % ((epoch+1), avg_cost, latent, reconstr, mean_params))

                if avg_cost < best_cost:
                    sys.stdout.write(" - Saving\n")
                    best_cost = avg_cost
                    self.saver.save(self.sess, "/tmp/model.ckpt")
                    saved = True
                    trials = 0
                else:
                    trials += 1
                    if trials < max_trials:
                        sys.stdout.write(" - Cost reversal trial %i\n" % trials)
                    else:
                        self.feed_dict[self.actual_learning_rate] *= 0.9
                        self.saver.restore(self.sess, "/tmp/model.ckpt")
                        trials = 0
                        sys.stdout.write(" - Reverting with learning rate (%f)\n" % self.feed_dict[self.actual_learning_rate])
  
        self.saver.restore(self.sess, "/tmp/model.ckpt")
        final_mean_params = np.mean(self.output("mp_mean_model"), axis=1)
        cost_history[training_epochs] = best_cost
        param_history[training_epochs, :] = final_mean_params
        return self, cost_history, param_history
