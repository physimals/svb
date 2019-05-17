"""
Stochastic Bayesian inference of a nonlinear model

Infers:
    - Posterior mean values of model parameters
    - A posterior covariance matrix

The general order for tensor dimensions is:
    - Voxel indexing (V=number of voxels)
    - Parameter indexing (P=number of parameters)
    - Sample indexing (N=number of samples)
    - Time point indexing (T=number of time points)

This ordering is chosen to allow the use of TensorFlow batch matrix
operations. However it is inconvenient for the model which would like
to be able to index input by parameter. For this reason we transpose
when calling the model's ``evaluate`` function.
"""
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import svb.noise as noise
from svb.prior import FactorisedPrior
from svb.posterior import MVNPosterior, FactorisedPosterior
from svb.utils import debug, LogBase

class SvbFit(LogBase):
    """
    Stochastic Bayesian model fitting

    :ivar model: Model instance to be fitted to some data
    :ivar prior: svb.prior.Prior instance defining the prior parameter distribution
    :ivar post: svb.posterior.Posterior instance defining the posterior parameter distribution
    :ivar params: Sequence of Parameter instances of parameters to infer. This includes the model
                  parameters and the noise parameter(s)
    """
    def __init__(self, model, **kwargs):
        LogBase.__init__(self)
        
        # The model to use for inference
        self.model = model

        # Debug mode
        self.debug = kwargs.get("debug", False)
        self.log.debug("Debug mode enabled")

        # All the parameters to infer - model parameters plus noise parameters
        self.params = list(model.params)
        self.noise = noise.NoiseParameter(debug=self.debug)
        self.params.append(self.noise)
        self._nparams = len(self.params)
        self._infer_covar = kwargs.get("infer_covar", False)
        self.mean_1, self.covar_1 = None, None

        # Set up the tensorflow graph which will be trained to do the inference
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Tensorflow input required for training
            #
            # x will have shape VxB where B is the batch size and V the number of voxels
            # xfull is the full data so will have shape VxT where N is the full time size
            # t will have shape 1xB or VxB depending on whether the timeseries is voxel
            # dependent (e.g. in 2D multi-slice readout)
            #
            # NB we don't know V, B and T at this stage so we set placeholder variables
            # self.nvoxels and self.nt and use validate_shape=False when creating
            # tensorflow Variables
            self.x = tf.placeholder(tf.float32, [None, None], name="x")
            self.xfull = self.log_tf(tf.placeholder(tf.float32, [None, None], name="xfull"))
            self.t = tf.placeholder(tf.float32, [None, None])
            self.learning_rate = tf.placeholder(tf.float32, shape=[])
            self.latent_weight = tf.placeholder(tf.float32, shape=[])
            #self.voxel_mask = tf.placeholder(tf.bamsool, [None])

            # Number of voxels in full data - known at runtime
            self.nvoxels = tf.shape(self.xfull)[0]

            # Number of time points in full data - known at runtime
            self.nt = tf.shape(self.xfull)[1]

            # Number of time points in each training batch - known at runtime
            self.batch_size = tf.shape(self.x)[1]

            # Number of samples per parameter for the sampling of the posterior distribution
            # Default to the batch size but not required to be the same
            self.num_samples = kwargs.get("num_samples", self.batch_size)

            # Create prior and posterior distributions
            param_priors = [param.voxelwise_prior(self.nvoxels) for param in self.params]
            param_posts = [param.voxelwise_posterior(self.t, self.xfull) for param in self.params]
            self.prior = FactorisedPrior(param_priors, name="prior", debug=self.debug)
            if self._infer_covar:
                self.post = MVNPosterior(param_posts, name="post", debug=self.debug)
            else:
                self.post = FactorisedPosterior(param_posts, name="post", debug=self.debug)

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

    def _get_model_prediction(self, samples):
        # Transform the underlying Gaussian samples into the values required by the model
        # This depends on each model parameter's underlying distribution
        # model_values has shape [P x V x D x 1]
        model_values = []
        for idx, param in enumerate(self.model.params):
            int_values = samples[:, idx, :]
            #int_values = tf.Print(int_values, [tf.shape(int_values)], "\nint_values")
            ext_values = tf.expand_dims(param.prior.transform.ext_values(int_values), -1)
            #ext_values = param.prior.transform.ext_values(int_values)
            #ext_values = tf.Print(ext_values, [tf.shape(ext_values)], "\next_values")
            model_values.append(ext_values)
        model_values = self.log_tf(tf.identity(model_values, name="model_values"), shape=True)

        # self.t has shape [V x B] or [1 x B]. Needs to be tiled
        # to [V x D x B] or [1 x D x B] to get a full prediction
        # for each sample
        #t = tf.tile(tf.reshape(self.t, [tf.shape(self.t)[0], 1, self.batch_size]), [1, self.num_samples, 1])
        t = self.log_tf(tf.identity(self.t, name="tvals"), shape=True)

        # Evaluate the model using the transformed values
        # Model prediction has shape [NV x D x B]
        return self.log_tf(tf.identity(self.model.evaluate(model_values, t), "model_predict"), shape=True)

    def _get_current_model_params(self):
        # Transform the underlying Gaussian samples into the values required by the model
        # This depends on each model parameter's underlying distribution
        # model_values has shape [P x V]
        model_values = []
        for idx, param in enumerate(self.params):
            int_values = self.post.mean[:, idx]
            model_values.append(param.post.transform.ext_values(int_values))
        self.model_params = self.log_tf(tf.identity(model_values, name="model_params"))

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
        # Generate a set of samples from the posterior [NV x P x B]
        samples = self.post.sample(self.num_samples)

        #samples = tf.boolean_mask(samples, self.voxel_mask)
        #data = tf.boolean_mask(self.x, self.voxel_mask)

        # Part 1: Calculate the log likelihood given our supplied sample values
        #
        # Note that we pass the total number of time points as we need to scale this term correctly
        # when the batch size is not the full data size

        # Unpack noise parameter. The noise model knows how to interpret this - typically it is the
        # log of a Gaussian variance but this is not required
        noise_samples = self.log_tf(tf.identity(samples[:, -1, :], name="noise_samples"))

        model_prediction = tf.identity(self._get_model_prediction(samples), name="modelfit")
        self._get_current_model_params()

        self.reconstr_loss = self.noise.log_likelihood(self.x, model_prediction, noise_samples, self.nt)
        self.reconstr_loss = self.log_tf(self.reconstr_loss)

        # Part 2: Latent loss = log (q(mp)/p(mp)) = log(q(mp)) - log(p(mp))
        latent_loss = self.post.entropy() - self.prior.mean_log_pdf(samples)
        self.latent_loss = tf.identity(latent_loss, name="latent_loss")
        self.latent_loss = self.log_tf(self.latent_loss)

        # Voxelwise cost
        self.cost = tf.add(self.reconstr_loss, self.latent_weight * self.latent_loss, name="cost")

        # Combine the costs from each voxel and use a single ADAM optimizer to optimize the mean cost
        self.mean_cost = tf.reduce_mean(self.cost, name="mean_cost")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimize = self.optimizer.minimize(self.mean_cost)

    def initialize(self):
        """
        Initialize global variables - i.e. initial values of posterior which
        may depend on the full data
        """
        self.sess.run(self.init, feed_dict=self.feed_dict)
        # Save the initial posterior
        #self.mean_1 = self.output("post_mean")
        #self.covar_1 = self.output("post_cov")

    def fit_batch(self):
        """
        Train model based on mini-batch of input data.

        :return: cost of mini-batch.
        """
        # Do the optimization (self.optimizer), but also calcuate the cost for reference
        # (self.cost, gives a second return argument)
        # Pass in X using the feed dictionary, as we want to process the batch we have been
        # provided X
        _, cost = self.sess.run([self.optimize, self.mean_cost], feed_dict=self.feed_dict)
        return cost

    def output(self, name):
        """
        Evaluate an output tensor

        e.g. ``output("mean")`` returns the current posterior means
        """
        return self.sess.run((self.sess.graph.get_tensor_by_name("%s:0" % name),), feed_dict=self.feed_dict)[0]

    def train(self, t, data, batch_size, training_epochs=100, display_step=1, output_graph=None,
              max_trials=50, quench_rate=0.5, fit_only_epochs=0, learning_rate=0.02,
              min_learning_rate=0.000001, tile_batches=1):
        """
        Train the graph to infer the posterior distribution given timeseries data
        """
        # Expect t to have a dimension for voxelwise variation even if it is the same for all voxels
        if t.ndim == 1:
            t = t.reshape(1, -1)

        n_voxels = data.shape[0]
        n_samples = t.shape[1]
        n_batches = int(np.ceil(float(n_samples) / batch_size))

        # Cost and parameter histories, mean and voxelwise
        cost_history = np.zeros([training_epochs+1])
        cost_history_v = np.zeros([n_voxels, training_epochs+1])
        param_history = np.zeros([training_epochs+1, self._nparams])
        param_history_v = np.zeros([n_voxels, training_epochs+1, self._nparams])
        #voxel_mask = np.ones([n_voxels], dtype=np.int)
        #voxel_mask[:1] = 1

        # Training cycle
        self.feed_dict = {
            self.xfull : data,
            self.learning_rate: learning_rate
        }
        self.initialize()
        if output_graph:
            writer = tf.summary.FileWriter(output_graph, self.sess.graph)

        trials, best_cost = 0, 1e12
        latent_weight = 0
        saved = False
        # multiple training epochs of gradient descent, i.e. make multiple 'passes' through the data.
        for epoch in range(training_epochs):
            err, avg_cost, avg_latent, avg_latent2, avg_reconstr, avg_zscore, avg_post_pdf, avg_prior_pdf = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            index = 0
            if epoch == fit_only_epochs:
                # Allow latent cost to have an impact on the fitting
                latent_weight = 1.0
                trials, best_cost = 0, 1e12

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

                    # If desired, repeat batch data to enable more samples
                    batch_xs = np.tile(batch_xs, (1, tile_batches))
                    t_xs = np.tile(t_xs, (1, tile_batches))

                    # Fit training using batch data
                    self.feed_dict[self.x] = batch_xs
                    self.feed_dict[self.t] = t_xs
                    self.feed_dict[self.latent_weight] = latent_weight
                    cost = np.mean(self.fit_batch())

                    # Compute costs
                    avg_cost += cost / n_batches
                    avg_latent += np.mean(self.output("latent_loss")) / n_batches
                    #avg_latent2 += np.mean(self.output("latent_loss2")) / n_batches
                    avg_reconstr += np.mean(self.output("reconstr_loss")) / n_batches
                    #avg_zscore += np.mean(self.output("square_diff")) / n_batches
                    #avg_prior_pdf += np.mean(self.output("prior_pdf")) / n_batches
                    #avg_post_pdf += np.mean(self.output("post_pdf")) / n_batches

            except tf.OpError:
                import traceback
                traceback.print_exc()
                err = 1

            params = self.output("model_params") # [P, V]
            var = self.output("post_var") # [V, P]
            mean_params = np.mean(params, axis=1)
            mean_var = np.mean(var, axis=0)
            if err or np.isnan(avg_cost) or np.any(np.isnan(mean_params)) or np.isnan(avg_latent) or np.isnan(avg_reconstr):
                self.feed_dict[self.learning_rate] = max(min_learning_rate,
                                                         self.feed_dict[self.learning_rate] * quench_rate)
                if saved:
                    self.saver.restore(self.sess, "/tmp/model.ckpt")
                sys.stdout.write("NaN values - reverting to previous best step with learning rate (%f)\n" % self.feed_dict[self.learning_rate])
            else:
                # Display logs per epoch step
                cost_history[epoch] = avg_cost
                cost_history_v[:, epoch] = self.output("cost")
                max_cost = np.max(cost_history_v[:, epoch])
                min_cost = np.min(cost_history_v[:, epoch])

                param_history[epoch, :] = mean_params
                param_history_v[:, epoch, :] = params.transpose()

                if epoch % display_step == 0:
                    sys.stdout.write("Epoch: %04d, mean cost=%f (latent=%f, reconstr=%f) mean params=%s mean_var=%s" % ((epoch+1), avg_cost, avg_latent, avg_reconstr, mean_params, mean_var))

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
                        self.feed_dict[self.learning_rate] = max(min_learning_rate,
                                                                 self.feed_dict[self.learning_rate] * quench_rate)
                        self.saver.restore(self.sess, "/tmp/model.ckpt")
                        trials = 0
                        sys.stdout.write(" - Reverting with learning rate (%f)\n" % self.feed_dict[self.learning_rate])

        self.saver.restore(self.sess, "/tmp/model.ckpt")
        self.feed_dict[self.x] = data
        self.feed_dict[self.t] = t
        sys.stdout.write("Best mean cost=%f / %f\n" % (best_cost, np.mean(self.output("cost"))))
        cost_history[training_epochs] = best_cost
        cost_history_v[:, training_epochs] = self.output("cost")

        params = self.output("model_params") # [P, NV]
        print("Final params: ", params)
        param_history[training_epochs, :] = np.mean(params, axis=1)
        param_history_v[:, training_epochs, :] = params.transpose()

        tiled_params = np.reshape(params, list(params.shape) + [1])
        fit = self.model.ievaluate(tiled_params, t)
        print(fit)
        return cost_history, param_history, cost_history_v, param_history_v, fit
