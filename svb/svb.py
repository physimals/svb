"""
Stochastic Bayesian inference of a nonlinear model

Infers:
    - Posterior mean values of model parameters
    - A posterior covariance matrix (which may be diagonal or a full
      positive-definite matrix)

The general order for tensor dimensions is:
    - Voxel indexing (V=number of voxels)
    - Parameter indexing (P=number of parameters)
    - Sample indexing (S=number of samples)
    - Data point indexing (B=batch size, i.e. number of time points
      being trained on, in some cases T=total number of time points
      in full data)

This ordering is chosen to allow the use of TensorFlow batch matrix
operations. However it is inconvenient for the model which would like
to be able to index input by parameter. For this reason we transpose
when calling the model's ``evaluate`` function to put the P dimension
first.
"""
import numpy as np
import tensorflow as tf

from .noise import NoiseParameter
from .prior import NormalPrior, FactorisedPrior, get_voxelwise_prior
from .posterior import NormalPosterior, FactorisedPosterior, MVNPosterior, get_voxelwise_posterior
from .utils import LogBase

class SvbFit(LogBase):
    """
    Stochastic Bayesian model fitting

    :ivar model: Model instance to be fitted to some data
    :ivar prior: svb.prior.Prior instance defining the prior parameter distribution
    :ivar post: svb.posterior.Posterior instance defining the posterior parameter distribution
    :ivar params: Sequence of Parameter instances of parameters to infer. This includes the model
                  parameters and the noise parameter(s)
    """
    def __init__(self, data_model, fwd_model, **kwargs):
        LogBase.__init__(self)

        # The data model
        self.data_model = data_model

        # The model to use for inference
        self.model = fwd_model

        # All the parameters to infer - model parameters plus noise parameters
        self.params = list(fwd_model.params)
        self.noise = NoiseParameter()
        self.params.append(self.noise)
        self._nparams = len(self.params)
        self._infer_covar = kwargs.get("infer_covar", False)
        self.mean_1, self.covar_1 = None, None

        # Set up the tensorflow graph which will be trained to do the inference
        self._graph = tf.Graph()
        with self._graph.as_default():
            # Create placeholder tensors to store the input data
            self._create_input_tensors()

            # Create voxelwise prior and posterior distribution tensors
            self._create_prior_post(**kwargs)

            # Define loss function based variational upper-bound and corresponding optimizer
            self._create_loss_optimizer()

            # Variable initializer
            self.init = tf.global_variables_initializer()

            # Tensorflow session for runnning graph
            self.sess = tf.Session()
    
    def _create_input_tensors(self):
        """
        Tensorflow input required for training
        
        x will have shape VxB where B is the batch size and V the number of voxels
        xfull is the full data so will have shape VxT where N is the full time size
        tpts_train will have shape 1xB or VxB depending on whether the timeseries is voxel
        dependent (e.g. in 2D multi-slice readout)
        
        NB we don't know V, B and T at this stage so we set placeholder variables
        self.nvoxels and self.nt_full and use validate_shape=False when creating
        tensorflow Variables
        """
        self.feed_dict = {}

        # Training data - may be mini-batch of full data
        self.data_train = tf.placeholder(tf.float32, [None, None], name="data_train")

        # Full data - we need this during training to correctly scale contributions
        # to the cost
        self.data_full = tf.placeholder(tf.float32, [None, None], name="data_full")

        # Time points in training data (not necessarily the full data)
        self.tpts_train = tf.placeholder(tf.float32, [None, None])

        # Learning rate - may be modified during training so must be a placeholder
        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # Amount of weight given to latent loss in cost function (0-1)
        self.latent_weight = tf.placeholder(tf.float32, shape=[])

        # Number of samples per parameter for the sampling of the posterior distribution
        self.num_samples = tf.placeholder(tf.int32, shape=[])

        # Number of voxels in full data - known at runtime
        self.nvoxels = tf.shape(self.data_full)[0]

        # Number of time points in each training batch - known at runtime
        self.batch_size = tf.shape(self.data_train)[1]

        # Number of time points in full data - known at runtime
        self.nt_full = tf.shape(self.data_full)[1]

        # Represent neighbour lists as sparse tensors
        self.nn = tf.SparseTensor(
            indices=self.data_model.indices_nn,
            values=np.ones((len(self.data_model.indices_nn),), dtype=np.float32),
            dense_shape=[self.data_model.n_unmasked_voxels, self.data_model.n_unmasked_voxels]
        )
        self.n2 = tf.SparseTensor(
            indices=self.data_model.indices_n2,
            values=np.ones((len(self.data_model.indices_n2),), dtype=np.float32),
            dense_shape=[self.data_model.n_unmasked_voxels, self.data_model.n_unmasked_voxels]
        )

    def _create_prior_post(self, **kwargs):
        """
        Create voxelwise prior and posterior distribution tensors
        """
        # Create posterior distribution - note this can be initialized using the actual data
        param_posts = []
        for idx, param in enumerate(self.params):        
            param_posts.append(get_voxelwise_posterior(param, self.tpts_train, self.data_full))

        if self._infer_covar:
            self.log.info("Inferring covariances (correlation) between Gaussian parameters")
            self.post = MVNPosterior(param_posts, name="post", **kwargs)
        else:
            self.log.info("Not inferring covariances between parameters")
            self.post = FactorisedPosterior(param_posts, name="post", **kwargs)

        # Create prior distribution - note this can make use of the posterior e.g.
        # for spatial regularization
        param_priors = []
        for idx, param in enumerate(self.params):            
            param_priors.append(get_voxelwise_prior(param, self.nvoxels, idx=idx, post=self.post, nn=self.nn, n2=self.n2))
        self.prior = FactorisedPrior(param_priors, name="prior", **kwargs)

        # If all of our priors and posteriors are Gaussian we can use an analytic expression for
        # the latent loss - so set this flag to decide if this is possible
        self.all_gaussian = (np.all([isinstance(prior, NormalPrior) for prior in param_priors]) and
                             np.all([isinstance(post, NormalPosterior) for post in param_posts]) and
                             not kwargs.get("force_num_latent_loss", False))
        if self.all_gaussian:
            self.log.info("Using analytical expression for latent loss since prior and posterior are Gaussian")
        else:
            self.log.info("Using numerical calculation of latent loss")

        # Report summary of parameters
        for idx, param in enumerate(self.params):
            self.log.info("%s: %s: %s" % (param, param_priors[idx], param_posts[idx]))

    def _get_model_prediction(self, samples):
        """
        Get a model prediction for the data batch being processed for each
        sample from the posterior

        :param samples: Tensor [V x P x S] containing samples from the posterior.
                        S is the number of samples (not always the same
                        as the batch size)

        :return Tensor [V x S x B]. B is the batch size, so for each voxel and sample
                we return a prediction which can be compared with the data batch
        """
        sample_params, mean_params = [], []
        for idx, param in enumerate(self.params):
            int_sample_params = samples[:, idx, :]
            int_mean_params = self.post.mean[:, idx]
            # Transform the underlying Gaussian samples into the values required by the model
            # This depends on each model parameter's underlying distribution
            #
            # The sample parameter values tensor also needs to be reshaped to [P x V x S x 1] so
            # the time values from the data batch will be broadcasted and a full prediction
            # returned for every sample
            sample_params.append(tf.expand_dims(param.post_dist.transform.ext_values(int_sample_params), -1))
            mean_params.append(param.post_dist.transform.ext_values(int_mean_params))

        sample_params = self.log_tf(tf.identity(sample_params, name="sample_params"), shape=True)
        mean_params = self.log_tf(tf.identity(mean_params, name="model_params"), shape=True)

        # The timepoints tensor has shape [V x B] or [1 x B]. It needs to be reshaped
        # to [V x 1 x B] or [1 x 1 x B] so it can be broadcast across each of the S samples
        tpts = tf.reshape(self.tpts_train, [tf.shape(self.tpts_train)[0], 1, self.batch_size])

        # Evaluate the model using the transformed values
        # Model prediction has shape [V x S x B]
        sample_prediction = self.log_tf(tf.identity(self.model.evaluate(sample_params, tpts),
                                                    "model_prediction"), shape=True)
        self.log_tf(tf.identity(self.model.evaluate(mean_params, tpts), "modelfit"), shape=True)
        return sample_prediction

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
        #data = tf.boolean_mask(self.data_train, self.voxel_mask)

        # Part 1: Reconstruction loss
        #
        # This deals with how well the parameters replicate the data and is defined as the
        # log-likelihood of the data (given the parameters).
        #
        # This is calculated from the noise model, as it boils down to how likely the deviations
        # from the model prediction to the data are within the noise model (with its current
        # parameters)

        # Unpack noise parameter. The noise model knows how to interpret this - typically it is the
        # log of a Gaussian variance but this is not required
        noise_samples = self.log_tf(tf.identity(samples[:, -1, :], name="noise_samples"))

        # Get the model prediction for the current set of parameters
        model_prediction = self._get_model_prediction(samples)

        # Note that we pass the total number of time points as we need to scale this term correctly
        # when the batch size is not the full data size
        reconstr_loss = self.noise.log_likelihood(self.data_train, model_prediction, noise_samples, self.nt_full)
        self.reconstr_loss = self.log_tf(tf.identity(reconstr_loss, name="reconstr_loss"))

        # Part 2: Latent loss
        #
        # This penalises parameters which are far from the prior
        # If both the prior and posterior are represented by an MVN we can calculate an analytic
        # expression for this cost. If not, we need to do it numerically using the posterior
        # sample obtained earlier. Note that the mean log pdf of the posterior based on sampling
        # from itself is just the distribution entropy so we allow it to be calculated without
        # sampling.
        if self.all_gaussian:
            latent_loss = tf.identity(self.post.latent_loss(self.prior), name="latent_loss")
        else:
            latent_loss = tf.subtract(self.post.entropy(samples), self.prior.mean_log_pdf(samples), name="latent_loss")

        self.latent_loss = self.log_tf(latent_loss)

        # Voxelwise cost is the sum of the latent and reconstruction cost but we have the possibility
        # of gradually introducing the latent loss via the latent_weight variable. This is based on
        # the theory that you should let the model fit the data first and then allow the fit to
        # be perturbed by the priors.
        if self.latent_weight == 0:
            self.cost = tf.identity(self.reconstr_loss, name="cost")
        else:
            self.cost = tf.add(self.reconstr_loss, self.latent_weight * self.latent_loss, name="cost")

        # Combine the costs from each voxel and use a single ADAM optimizer to optimize the mean cost
        # It is also possible to optimize the total cost but this makes it harder to compare with
        # variable numbers of voxels
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

        :return: Tuple of total cost of mini-batch, latent cost and reconstruction cost
        """
        _, cost, latent, reconstr = self.sess.run([self.optimize,
                                                   self.cost,
                                                   self.latent_loss,
                                                   self.reconstr_loss], feed_dict=self.feed_dict)
        return cost, latent, reconstr

    def output(self, name):
        """
        Evaluate an output tensor

        e.g. ``output("mean")`` returns the current posterior means
        """
        return self.sess.run((self.sess.graph.get_tensor_by_name("%s:0" % name),), feed_dict=self.feed_dict)[0]

    def state(self):
        """
        Get the current state of the optimization.

        This can be used to restart from a previous state if a numerical error occurs
        """
        tensors = self.post.state()
        return self.sess.run(tensors, feed_dict=self.feed_dict)

    def set_state(self, state):
        """
        Set the state of the optimization

        :param state: State as returned by the ``state()`` method
        """
        ops = self.post.set_state(state)
        self.sess.run(ops, feed_dict=self.feed_dict)

    def train(self, tpts, data,
              batch_size=None, sequential_batches=False,
              epochs=100, fit_only_epochs=0, display_step=1,
              learning_rate=0.02, lr_quench=0.5, max_trials=50, lr_min=0.000001, revert_post=True,
              **kwargs):
        """
        Train the graph to infer the posterior distribution given timeseries data

        :param tpts: Time series values. Should have shape [T] or [V, T] depending on whether timeseries is
                  constant or varies voxelwise
        :param data: Full timeseries data, shape [V, T]

        Optional arguments:

        :param batch_size: Batch size to use when training model. Need not be a factor of T, however if not
                           batches will not all be the same size. If not specified, data size is used (i.e.
                           no mini-batch optimization)
        :param training epochs: Number of training epochs
        :param fit_only_epochs: If specified, this number of epochs will be restricted to fitting only
                                and ignore prior information. In practice this means only the
                                reconstruction loss is considered not the latent cost
        :param display_step: How many steps to execute for each display line
        :param max_trials: How many epoch to continue for without an improvement in the mean cost before
                           adjusting the learning rate
        :param learning_rate: Initial learning rate
        :param min_learning_rate: Minimum learning rate - if the learning rate is adjusted it will never
                                  go below this value
        :param quench_rate: When adjusting the learning rate, the factor to reduce it by
        :param sample_size: Number of samples to use when estimating expectations over the posterior
        """
        # Expect tpts to have a dimension for voxelwise variation even if it is the same for all voxels
        if tpts.ndim == 1:
            tpts = tpts.reshape(1, -1)

        # Determine number of voxels and timepoints and check consistent
        n_voxels, n_timepoints = tuple(data.shape)
        if tpts.shape[0] > 1 and tpts.shape[0] != n_voxels:
            raise ValueError("Time points has %i voxels, but data has %i" % (tpts.shape[0], n_voxels))
        if tpts.shape[1] != n_timepoints:
            raise ValueError("Time points has length %i, but data has %i volumes" % (tpts.shape[1], n_timepoints))

        # Determine number of batches and sample size
        if batch_size is None:
            batch_size = n_timepoints
        n_batches = int(np.ceil(float(n_timepoints) / batch_size))
        num_samples = kwargs.get("sample_size", batch_size)

        # Cost and parameter histories, mean and voxelwise
        mean_cost_history = np.zeros([epochs+1])
        voxel_cost_history = np.zeros([n_voxels, epochs+1])
        mean_param_history = np.zeros([epochs+1, self._nparams])
        voxel_param_history = np.zeros([n_voxels, epochs+1, self._nparams])

        # Training cycle
        self.feed_dict = {
            self.data_full : data,
            self.learning_rate: learning_rate,
            self.num_samples: num_samples,
        }
        self.initialize()

        trials, best_cost, best_state = 0, 1e12, None
        latent_weight = 0

        # Each epoch passes through the whole data but it may do this in 'batches' so there may be
        # multiple training iterations per epoch, one for each batch
        for epoch in range(epochs):
            try:
                total_cost = np.zeros([n_voxels])
                total_latent = np.zeros([n_voxels])
                total_reconstr = np.zeros([n_voxels])
                err, index = False, 0

                if epoch == fit_only_epochs:
                    # Once we have completed fit_only_epochs of training we will allow the latent cost to have
                    # an impact and reset the best cost accordingly. By default this happens on the first epoch
                    latent_weight = 1.0
                    trials, best_cost = 0, 1e12

                # Iterate over training batches - note that there may be only one
                for i in range(n_batches):
                    #print(i)
                    if sequential_batches:
                        # Batches are defined by sequential data time points
                        if i == n_batches - 1:
                            # Batch size may not be an exact factor of the number of time points
                            # so make the last batch the right size so all of the data is used
                            batch_size += n_timepoints - n_batches * batch_size
                        batch_data = data[:, index:index+batch_size]
                        batch_tpts = tpts[:, index:index+batch_size]
                        index = index + batch_size
                    else:
                        # Batches are defined by constant strides through the data time points
                        # This automatically handles case where number of time point does not
                        # exactly divide into batches
                        batch_data = data[:, i::n_batches]
                        batch_tpts = tpts[:, i::n_batches]

                    # Perform a training iteration using batch data
                    self.feed_dict.update({
                        self.data_train: batch_data,
                        self.tpts_train : batch_tpts,
                        self.latent_weight : latent_weight
                    })
                    batch_cost, batch_latent, batch_reconstr = self.fit_batch()

                    # Add contribution to mean cost over all batches (i.e. the whole data set)
                    # Note that batches are equally weighted which might not be quite right
                    # if they are different sizes
                    total_cost += batch_cost / n_batches
                    total_latent += batch_latent / n_batches
                    total_reconstr += batch_reconstr / n_batches

            except tf.OpError:
                self.log.exception("Numerical error fitting batch")
                err = True

            # Record the cost and parameter values at the end of each epoch. We do
            # this voxelwise and the mean over voxels
            params = self.output("model_params") # [P, V]
            var = self.output("post_var") # [V, P]
            mean_params = np.mean(params, axis=1)
            mean_var = np.mean(var, axis=0)

            mean_total_cost = np.mean(total_cost)
            mean_total_latent = np.mean(total_latent)
            mean_total_reconst = np.mean(total_reconstr)

            mean_cost_history[epoch] = mean_total_cost
            voxel_cost_history[:, epoch] = total_cost
            mean_param_history[epoch, :] = mean_params
            voxel_param_history[:, epoch, :] = params.transpose()

            if err or np.isnan(mean_total_cost) or np.any(np.isnan(mean_params)):
                # Numerical errors while processing this epoch. We will reduce the learning rate
                # if possible and revert to best previously saved params
                self.feed_dict[self.learning_rate] = max(lr_min, self.feed_dict[self.learning_rate] * lr_quench)
                self.initialize()
                if best_state is not None:
                    self.set_state(best_state)
                    params = self.output("model_params") # [P, V]
                    var = self.output("post_var") # [V, P]
                    mean_params = np.mean(params, axis=1)
                    mean_var = np.mean(var, axis=0)
                outcome = "Revert -> LR=%f" % self.feed_dict[self.learning_rate]
                self.log.warning("Numerical errors: Revert with learning rate: %f" % self.feed_dict[self.learning_rate])
            elif mean_total_cost < best_cost:
                # There was an improvement in the mean cost - save the current state of the posterior
                outcome = "Saving"
                best_cost = mean_total_cost
                best_state = self.state()
                trials = 0
            else:
                # The mean cost did not improve. We will continue until it has not improved for max_trials
                # epochs and then revert with lower learning rate
                trials += 1
                if trials < max_trials:
                    outcome = "Trial %i" % trials
                else:
                    self.feed_dict[self.learning_rate] = max(lr_min, self.feed_dict[self.learning_rate] * lr_quench)
                    if revert_post:
                        self.set_state(best_state)
                        outcome = "Revert -> LR=%f" % self.feed_dict[self.learning_rate]
                    else:
                        outcome = "Continue -> LR=%f" % self.feed_dict[self.learning_rate]
                    trials = 0

            if epoch % display_step == 0:
                state_str = "mean cost=%f (latent=%f, reconstr=%f) mean params=%s mean_var=%s" % (
                    mean_total_cost, mean_total_latent, mean_total_reconst, mean_params, mean_var)
                self.log.info("Epoch %04d: %s - %s", (epoch+1), state_str, outcome)
                
        # At the end of training we revert to the state with best mean cost and write a final history step
        # with these values. Note that the cost may not be as reported earlier as this was based on a
        # mean over the training batches whereas here we recalculate the cost for the whole data set.
        self.set_state(best_state)
        self.feed_dict[self.data_train] = data
        self.feed_dict[self.tpts_train] = tpts
        cost = self.output("cost") # [V]
        params = self.output("model_params") # [P, NV]
        self.log.info("Best mean cost=%f / %f\n", best_cost, np.mean(cost))
        self.log.info("Final params: %s", np.mean(params, axis=1))

        cost = self.output("cost") # [V]
        params = self.output("model_params") # [P, NV]
        self.log.info("Best mean cost=%f / %f\n", best_cost, np.mean(cost))
        self.log.info("Final params: %s", np.mean(params, axis=1))

        mean_cost_history[-1] = best_cost
        mean_param_history[-1, :] = np.mean(params, axis=1)
        voxel_cost_history[:, -1] = cost
        voxel_param_history[:, -1, :] = params.transpose()

        # Return cost and parameter history and model fit
        tiled_params = np.reshape(params, list(params.shape) + [1])
        fit = self.model.ievaluate(tiled_params, tpts)
        return mean_cost_history, mean_param_history, voxel_cost_history, voxel_param_history, fit

    def modelfit(self, model_params, tpts):
        """
        Get the model fit using the current mean model parametes
        """
        model_params = self.output("model_params")
        model_params = np.reshape(model_params, list(model_params.shape) + [1])
        return self.model.ievaluate(model_params, tpts)
