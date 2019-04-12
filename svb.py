"""
Stochastic Bayesian inference of a nonlinear model
"""
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import dist
import noise

class SvbFit(object):
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
        self.params = list(model.params)
        self.noise = noise.NoiseParameter()
        self.params.append(self.noise)
        self.nparams = len(self.params)

        # Debug mode
        self.debug = kwargs.get("debug", False)

        # Set up the tensorflow graph which will be trained to do the inference
        self.graph = tf.Graph()
        with self.graph.as_default():
            #tf.set_random_seed(1)

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
            #self.voxel_mask = tf.placeholder(tf.bool, [None])
         
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
            self._init_posterior()
            
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
        mean = np.array([param.prior.nmean for param in self.params])
        var = np.array([param.prior.nvar for param in self.params])
        #print("Prior variance: ", var)
        mean_0 = tf.constant(mean, dtype=tf.float32, name="mean_0")
        var_0 = tf.constant(var, dtype=tf.float32, name="var_0")
        #if self.debug:
        #    mean_0 = tf.Print(mean_0, [mean_0], "\nPrior mean", summarize=100)
        #    cov_0 = tf.Print(cov_0, [cov_0], "\nPrior variance", summarize=100)

        voxelwise_mean_0 = tf.tile(tf.reshape(mean_0, [1, self.nparams]), [self.nvoxels, 1])
        voxelwise_var_0 = tf.tile(tf.reshape(var_0, [1, self.nparams]), [self.nvoxels, 1])
        self.prior = dist.ConstantMVN(self.params, mean_init=voxelwise_mean_0, var_init=voxelwise_var_0, name="prior", corr=False, debug=self.debug)

    def _init_posterior(self):
        initial_mean = [param.initial(self.t, self.xfull)[0] for param in self.params]
        initial_var = [param.initial(self.t, self.xfull)[1] for param in self.params]

        initial_mean = tf.transpose(initial_mean, [1, 0])
        initial_var = tf.transpose(initial_var, [1, 0])
        initial_var = tf.Print(initial_var, [initial_var], "initial var", summarize=100)
        self.post = dist.MVN(self.params, mean_init=initial_mean, var_init=initial_var, name="post", corr=self.mode_corr == "infer_post_corr", debug=self.debug)

    def _get_model_prediction(self, samples):
        # Transform the underlying Gaussian samples into the values required by the model
        # This depends on each model parameter's underlying distribution
        model_values = []
        for idx, param in enumerate(self.model.params):
            model_values.append(param.prior.tomodel(samples[:, idx, :]))

        # Evaluate the model using the transformed values
        return self.model.evaluate(model_values, self.t)

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
        # Generate a set of samples from the posterior
        samples = self.post.sample(self.batch_size)

        #samples = tf.boolean_mask(samples, self.voxel_mask)
        #data = tf.boolean_mask(self.x, self.voxel_mask)

        # Part 1: Calculate the log likelihood given our supplied sample values
        #
        # Note that we pass the total number of time points as we need to scale this term correctly
        # when the batch size is not the full data size
        
        # Unpack noise parameter. The noise model knows how to interpret this - typically it is the
        # log of a Gaussian variance but this is not required
        noise = samples[:, -1, :]

        # Since we are processing only a batch of the data at a time, we need to scale this term
        # correctly to the latent loss
        scale = tf.to_float(self.nt) / tf.to_float(self.batch_size)
        #scale = 1.0

        model_prediction = self._get_model_prediction(samples)
        self.reconstr_loss = tf.multiply(scale, self.noise.log_likelihood(self.x, model_prediction, noise), name="reconstr_loss")
        if self.debug:
            self.reconstr_loss = tf.Print(self.reconstr_loss, [tf.shape(self.reconstr_loss), self.reconstr_loss], "\nreconstr", summarize=100)

        # Part 1: Latent loss
        #
        # log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        latent_loss = self.post.mean_log_pdf(samples) - self.prior.mean_log_pdf(samples)
        self.latent_loss = tf.identity(latent_loss, name="latent_loss")

        #inv_mp_covar_0 = tf.matrix_inverse(self.prior.cov)
        #inv_mp_covar_0 = tf.tile(tf.reshape(inv_mp_covar_0, (1, self.nparams, self.nparams)), (self.nvoxels, 1, 1))
        #mn = tf.subtract(self.post.mean, tf.reshape(self.prior.mean, (1, -1)))
        #t1 = tf.trace(tf.matmul(inv_mp_covar_0, self.post.cov))
        #t2 = tf.matmul(tf.reshape(mn, (self.nvoxels, 1, -1)), inv_mp_covar_0)
        #t3 = tf.reshape(tf.matmul(t2, tf.reshape(mn, (self.nvoxels, -1, 1))), [self.nvoxels])
        #t4 = tf.log(tf.matrix_determinant(self.prior.cov, name='det_mp_covar_0'))
        #t5 = tf.log(tf.matrix_determinant(self.post.cov, name='det_mp_covar'))
        ##t5 = tf.log(tf.matrix_determinant(self.mp_covar + self.reg_cov, name='det_mp_covar'))
        #latent_loss = 0.5*(t1 + t3 - self.nparams + t4 - t5)
        #self.latent_loss = tf.identity(latent_loss, name="latent_loss")

        if self.debug:
            self.latent_loss = tf.Print(self.latent_loss, [tf.shape(self.latent_loss), self.latent_loss], "\nlatent", summarize=100)

        # Sum the cost from each voxel and use a single optimizer
        self.cost = tf.add(self.reconstr_loss, self.latent_loss, name="cost")

        self.mean_cost = tf.reduce_mean(self.cost, name="mean_cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.actual_learning_rate)
        
        self.optimize = self.optimizer.minimize(self.mean_cost)

    def initialize(self):
        """
        Initialize global variables - i.e. initial values of posterior which
        may depend on the full data
        """
        self.sess.run(self.init, feed_dict=self.feed_dict)
        # Save the initial posterior
        self.mean_1 = self.output("post_mean", feed_dict=self.feed_dict)
        self.covar_1 = self.output("post_cov", feed_dict=self.feed_dict)

    def fit_batch(self):
        """
        Train model based on mini-batch of input data.       

        :return: cost of mini-batch.
        """
        # Do the optimization (self.optimizer), but also calcuate the cost for reference (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been provided X
        _, cost = self.sess.run([self.optimize, self.mean_cost], feed_dict=self.feed_dict)
        return cost

    def output(self, name, feed_dict=None):
        """
        Evaluate an output tensor

        e.g. ``output("mean")`` returns the current posterior means
        """
        return self.sess.run((self.sess.graph.get_tensor_by_name("%s:0" % name),), feed_dict=self.feed_dict)[0]

    def train(self, t, data, batch_size, training_epochs=100, display_step=1, output_graph=None, max_trials=50, quench_rate=0.5):
        """
        Train the graph to infer the posterior distribution given timeseries data 
        """
        # Expect t to have a dimension for voxelwise variation even if it is the same for all voxels
        if t.ndim == 1:
            t = t.reshape(1, -1)
        n_voxels = data.shape[0]
        n_samples = t.shape[1]
        n_batches = int(np.ceil(float(n_samples) / batch_size))
        cost_history=np.zeros([training_epochs+1]) 
        cost_history_v=np.zeros([n_voxels, training_epochs+1]) 
        param_history=np.zeros([training_epochs+1, self.nparams]) 
        voxel_mask = np.ones([n_voxels])
        #voxel_mask[:1] = 1

        # Training cycle
        self.feed_dict={
            self.xfull : data, 
            self.actual_learning_rate: self.learning_rate
        }
        self.initialize()
        if output_graph:
            writer = tf.summary.FileWriter(output_graph, self.sess.graph)
        
        print("Initial: ", self.output("post_mean"))
        print("Initial: ", self.output("post_cov"))
        trials, best_cost = 0, 1e12
        saved = False
        for epoch in range(training_epochs): # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.
            err, avg_cost, avg_latent, avg_reconstr = 0, 0.0, 0.0, 0.0
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
                    #self.feed_dict[self.voxel_mask] = voxel_mask
                    cost = np.mean(self.fit_batch())
                                
                    # Compute average cost
                    #print("batch cost: ", cost)
                    avg_cost += cost / n_batches
                    avg_latent += np.mean(self.output("latent_loss")) / n_batches
                    avg_reconstr += np.mean(self.output("reconstr_loss")) / n_batches
                    #print("batch cost: ", self.output("cost"), self.output("latent_loss"), self.output("reconstr_loss"))
                    
            except tf.OpError:
                import traceback
                traceback.print_exc()
                err = 1

            mean_params = np.mean(self.output("post_mean_model"), axis=1)
            if err or np.isnan(avg_cost) or np.any(np.isnan(mean_params)) or np.isnan(avg_latent) or np.isnan(avg_reconstr):
                self.feed_dict[self.actual_learning_rate] *= quench_rate
                if saved:
                    self.saver.restore(self.sess, "/tmp/model.ckpt")
                sys.stdout.write("NaN values - reverting to previous best step with lower learning rate (%f)\n" % self.feed_dict[self.actual_learning_rate])
            else:
                # Display logs per epoch step
                cost_history[epoch] = avg_cost
                cost_history_v[:, epoch] = self.output("cost")
                param_history[epoch, :] = mean_params
                if epoch % display_step == 0:
                    sys.stdout.write("Epoch: %04d, mean cost=%f (%f, %f), mean params=%s" % ((epoch+1), avg_cost, avg_latent, avg_reconstr, mean_params))

                if avg_cost < best_cost:
                    sys.stdout.write(" - Saving\n")
                    best_cost = avg_cost
                    self.saver.save(self.sess, "/tmp/model.ckpt")
                    saved = True
                    trials = 0
                else:
                    trials += 1
                    if trials < max_trials:
                        sys.stdout.write(" - trial %i\n" % trials)
                    else:
                        self.feed_dict[self.actual_learning_rate] *= quench_rate
                        self.saver.restore(self.sess, "/tmp/model.ckpt")
                        trials = 0
                        sys.stdout.write(" - Reverting with learning rate (%f)\n" % self.feed_dict[self.actual_learning_rate])
  
            #sys.exit(0)

        self.saver.restore(self.sess, "/tmp/model.ckpt")
        sys.stdout.write("Best mean cost=%f\n" % best_cost)
        final_mean_params = np.mean(self.output("post_mean_model"), axis=1)
        cost_history[training_epochs] = best_cost
        param_history[training_epochs, :] = final_mean_params
        return self, cost_history, param_history, cost_history_v
