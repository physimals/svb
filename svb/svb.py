"""
Stochastic Bayesian inference of a nonlinear model

Infers:
    - Posterior mean values of model parameters
    - A posterior covariance matrix (which may be diagonal or a full
      positive-definite matrix)

The general order for tensor dimensions is:
    - Voxel indexing (V=number of voxels / W=number of parameter nodes)
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

The parameter nodes, W, are the set of points on which parameters are defined
and will be output. They may be voxel centres, or surface element nodes. The
data voxels, V, on the other hand are the points on which the data to be fitted to
is defined. Typically this will be volumetric voxels as that is what most
imaging experiments output as raw data.

In many cases, W will be the same as V since we are inferring volumetric parameter
maps from volumetric data. However we might alternatively want to infer surface
based parameter maps but keep the comparison to the measured volumetric data. In
this case V and W will be different. The key point at which this difference is handled
is the model evaluation which takes parameters defined on W and outputs a prediction
defined on V.

V and W are currently identical but may not be in the future. For example
we may want to estimate parameters on a surface (W=number of surface 
nodes) using data defined on a volume (V=number of voxels).

Ideas for per voxel/vertex convergence:

    - Maintain vertex_mask as member. Initially all ones
    - Mask nodes when generating samples and evaluating model. The
      latent cost will be over unmasked nodes only.
    - PROBLEM: need reconstruction cost defined over full voxel set
      hence need to project model evaluation onto all voxels. So
      masked nodes still need to keep their previous model evaluation
      output
    - Define criteria for masking nodes after each epoch
    - PROBLEM: spatial interactions make per-voxel convergence difficult.
      Maybe only do full set convergence in this case (like Fabber)
"""
import time
import six

import numpy as np
from scipy import sparse
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

from .noise import NoiseParameter
from .prior import NormalPrior, FactorisedPrior, get_prior
from .posterior import NormalPosterior, FactorisedPosterior, MVNPosterior, get_posterior
from .utils import LogBase, TF_DTYPE, NP_DTYPE

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
        self.data_train = tf.placeholder(TF_DTYPE, [None, None], name="data_train")

        # Time points in training data (not necessarily the full data - may be mini-batch)
        self.tpts_train = tf.placeholder(TF_DTYPE, [None, None])

        # Full data - we need this during training to correctly scale contributions
        # to the cost
        #self.data_full = tf.placeholder(TF_DTYPE, [None, None], name="data_full")
        self.data_full = tf.constant(self.data_model.data_flattened, dtype=TF_DTYPE, name="data_full")

        # Number of time points in full data - known at runtime
        #self.nt_full = tf.shape(self.data_full)[1]
        self.nt_full = self.data_model.n_tpts

        # Initial learning rate
        self.initial_lr = tf.placeholder(TF_DTYPE, shape=[])

        # Counters to keep track of how far through the full set of optimization steps
        # we have reached
        self.global_step = tf.train.create_global_step()
        self.num_steps = tf.placeholder(TF_DTYPE, shape=[])

        # Optional learning rate decay - to disable simply set decay rate to 1.0
        self.lr_decay_rate = tf.placeholder(TF_DTYPE, shape=[])
        self.learning_rate = tf.train.exponential_decay(
            self.initial_lr,
            self.global_step,
            self.num_steps,
            self.lr_decay_rate,
            staircase=False,
        )

        # Amount of weight given to latent loss in cost function (0-1)
        self.latent_weight = tf.placeholder(TF_DTYPE, shape=[])

        # Initial number of samples per parameter for the sampling of the posterior distribution
        self.initial_ss = tf.placeholder(tf.int32, shape=[])

        # Optional increase in the sample size - to disable set factor to 1.0
        self.ss_increase_factor = tf.placeholder(TF_DTYPE, shape=[])
        self.sample_size = tf.cast(tf.round(tf.train.exponential_decay(
            tf.to_float(self.initial_ss),
            self.global_step,
            self.num_steps,
            self.ss_increase_factor,
            staircase=False,
            #tf.to_float(self.initial_ss) * self.ss_increase_factor,
            #power=1.0,
        )), tf.int32)

        # Number of voxels in full data (V) - known at runtime
        #self.nvoxels = tf.shape(self.data_full)[0]
        self.nvoxels = self.data_model.n_unmasked_voxels

        # Number of parameter nodes (W) - known at runtime. Currently equal
        # to number of voxels. In future this will be defined by the data model.
        # TODO: update the above comment 
        self.nnodes = self.data_model.n_nodes
 


    def _create_prior_post(self, **kwargs):
        """
        Create voxelwise prior and posterior distribution tensors
        """
        self.log.info("Setting up prior and posterior")
        # Create posterior distribution - note this can be initialized using the actual data
        gaussian_posts, nongaussian_posts, all_posts = [], [], []
        for idx, param in enumerate(self.params):    
            post = get_posterior(idx, param, self.tpts_train, self.data_model, init=self.data_model.post_init, **kwargs)
            if isinstance(post, NormalPosterior):
                gaussian_posts.append(post)
                # FIXME Noise parameter hack
                if idx == len(self.params) - 1:
                    noise_gaussian = True
            else:
                nongaussian_posts.append(post)
                if idx == len(self.params) - 1:
                    noise_gaussian = False
            all_posts.append(post)

        if self._infer_covar:
            self.log.info(" - Inferring covariances (correlation) between %i Gaussian parameters" % len(gaussian_posts))
            if nongaussian_posts:
                self.log.info(" - Adding %i non-Gaussian parameters" % len(nongaussian_posts))
                self.post = FactorisedPosterior([MVNPosterior(gaussian_posts, **kwargs)] + nongaussian_posts, name="post", **kwargs)
            else:
                self.post = MVNPosterior(gaussian_posts, name="post", init=self.data_model.post_init, **kwargs)

            # Depending on whether the noise is gaussian or not it may appear in 
            # a different position in the parameter lists
            if noise_gaussian:
                self.noise_idx = len(all_posts) - len(nongaussian_posts) - 1
            else:
                self.noise_idx = len(all_posts) - 1
        else:
            self.log.info(" - Not inferring covariances between parameters")
            self.post = FactorisedPosterior(all_posts, name="post", **kwargs)
            self.noise_idx = len(all_posts) - 1

        # Create prior distribution - note this can make use of the posterior e.g.
        # for spatial regularization
        all_priors = []
        for idx, param in enumerate(self.params):            
            all_priors.append(get_prior(param, self.data_model, idx=idx, post=self.post))
        self.prior = FactorisedPrior(all_priors, name="prior", **kwargs)

        # If all of our priors and posteriors are Gaussian we can use an analytic expression for
        # the latent loss - so set this flag to decide if this is possible
        self.analytic_latent_loss = (np.all([isinstance(prior, NormalPrior) for prior in all_priors]) and
                             not nongaussian_posts and not kwargs.get("force_num_latent_loss", False))
        if self.analytic_latent_loss:
            self.log.info(" - Using analytical expression for latent loss since prior and posterior are Gaussian")
        else:
            self.log.info(" - Using numerical calculation of latent loss")

        # Report summary of parameters
        for idx, param in enumerate(self.params):
            self.log.info(" - %s", param)
            self.log.info("   - Prior: %s %s", param.prior_dist, all_priors[idx])
            self.log.info("   - Posterior: %s %s", param.post_dist, all_posts[idx])

    def _get_model_prediction(self, param_samples, noise_samples):
        """
        Get a model prediction for the data batch being processed for each
        sample from the posterior

        FIXME assuming noise is last parameter

        :param samples: Tensor [W x P x S] containing samples from the posterior.
                        S is the number of samples (not always the same
                        as the batch size)

        :return Tensor [V x S x B]. B is the batch size, so for each voxel and sample
                we return a prediction which can be compared with the data batch
        """
        model_samples, model_means, model_vars = [], [], []
        for idx, param in zip(range(param_samples.shape[1]), self.params):
            assert idx != self.noise_idx, 'Iterating over noise parameter'
            int_samples = param_samples[:, idx, :]
            int_means = self.post.mean[:, idx]
            int_vars = self.post.var[:, idx]
            
            # Transform the underlying Gaussian samples into the values required by the model
            # This depends on each model parameter's underlying distribution
            #
            # The sample parameter values tensor also needs to be reshaped to [P x W x S x 1] so
            # the time values from the data batch will be broadcasted and a full prediction
            # returned for every sample
            model_samples.append(tf.expand_dims(param.post_dist.transform.ext_values(int_samples), -1))
            ext_means, ext_vars = param.post_dist.transform.ext_moments(int_means, int_vars)
            model_means.append(ext_means)
            model_vars.append(ext_vars)

        # Produce a noise prediction 
        noise_param = self.params[self.noise_idx]
        int_means = self.post.mean[:, self.noise_idx]
        int_vars = self.post.var[:, self.noise_idx]
        self.noise_prediction = tf.expand_dims(
            noise_param.post_dist.transform.ext_values(noise_samples), -1)
        ext_means, ext_vars = noise_param.post_dist.transform.ext_moments(int_means, int_vars)
        model_means.append(ext_means)
        model_vars.append(ext_vars)
        self.noise_mean = self.log_tf(tf.identity(ext_means), name="noise_mean")
        self.noise_var = self.log_tf(tf.identity(ext_vars), name="noise_vars")
        
        # Produce the model prediction 
        # Define convenience tensors for querying the model-space sample, means and prediction
        # modelfit_nodes has shape [W x B]
        self.model_samples = self.log_tf(tf.identity(model_samples, name="model_samples"))
        self.model_means = self.log_tf(tf.identity(model_means, name="model_means"))
        self.model_vars = self.log_tf(tf.identity(model_vars, name="model_vars"))
        self.modelfit_nodes = self.log_tf(tf.identity(self.model.evaluate(tf.expand_dims(self.model_means, -1), self.tpts_train), "modelfit_nodes"))
        
        # FIXME compatibility
        self.modelfit = self.log_tf(self.modelfit_nodes, name="modelfit")

        # The timepoints tensor has shape [V x B] or [1 x B]. It needs to be reshaped
        # to [V x 1 x B] or [1 x 1 x B] so it can be broadcast across each of the S samples
        sample_tpts = self.log_tf(tf.expand_dims(self.tpts_train, 1), name="sample_tpts")

        # Evaluate the model using the transformed values
        # Model prediction has shape [W x S x B]
        self.sample_predictions = self.log_tf(tf.identity(self.model.evaluate(model_samples, sample_tpts),
                                                          "sample_predictions"), shape=True, force=False)
        return self.sample_predictions, self.noise_prediction

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
        # Generate a set of samples from the posterior [W x P x B]
        samples = self.log_tf(self.post.sample(self.sample_size), name="samples", shape=True, force=False)
        noise_samples = samples[:,self.noise_idx,:]
        param_samples = samples[:,:self.noise_idx,:]

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

        # Get the model prediction for the current set of parameter samples
        # Model prediction has shape [W x S x B]
        model_prediction, noise_prediction = self._get_model_prediction(
                                                        param_samples, noise_samples)



        # Unpack noise parameter. The noise model knows how to interpret this - typically it is the
        # log of a Gaussian variance but this is not required
        #noise_samples = self.log_tf(tf.identity(samples[:, self.noise_idx, :], name="noise_samples"), shape=True, force=True)
        # noise_samples = self.log_tf(tf.squeeze(self.model_samples[-1]), name="noise_samples", shape=True, force=False)
        noise_samples = tf.squeeze(noise_prediction)

        # Note that we pass the total number of time points as we need to scale this term correctly
        # when the batch size is not the full data size
        model_prediction_voxels = self.log_tf(self.data_model.nodes_to_voxels_ts(model_prediction), name="model_prediction_voxels", shape=True, force=False)
        noise_samples_voxels = self.log_tf(self.data_model.nodes_to_voxels(noise_samples), name="noise_samples_voxels", shape=True, force=False)
        reconstr_loss = self.noise.log_likelihood(self.data_train, model_prediction_voxels, noise_samples_voxels, self.nt_full)
        self.reconstr_loss = self.log_tf(tf.identity(reconstr_loss, name="reconstr_loss"), shape=True, force=False)

        # Part 2: Latent loss
        #
        # This penalises parameters which are far from the prior
        # If both the prior and posterior are represented by an MVN we can calculate an analytic
        # expression for this cost. If not, we need to do it numerically using the posterior
        # sample obtained earlier. Note that the mean log pdf of the posterior based on sampling
        # from itself is just the distribution entropy so we allow it to be calculated without
        # sampling.
        if self.analytic_latent_loss:
            latent_loss = tf.identity(self.post.latent_loss(self.prior), name="latent_loss")
        else:
            latent_loss = tf.subtract(self.post.entropy(samples), self.prior.mean_log_pdf(samples), name="latent_loss")

        self.latent_loss = self.log_tf(latent_loss)

        # Voxelwise cost is the sum of the latent and reconstruction cost but we have the possibility
        # of gradually introducing the latent loss via the latent_weight variable. This is based on
        # the theory that you should let the model fit the data first and then allow the fit to
        # be perturbed by the priors.
        if self.latent_weight == 0:
            #self.cost = tf.identity(self.reconstr_loss, name="cost")
            raise NotImplementedError()
        else:
            # FIXME can't add reconstr and latent costs as one defined on voxels the other on nodes
            #self.cost = tf.add(self.reconstr_loss, self.latent_weight * self.latent_loss, name="cost")
            self.cost = self.log_tf(tf.identity(self.reconstr_loss, name="cost"), force=False, shape=True)
            self.mean_reconstr_cost = tf.reduce_mean(self.reconstr_loss, name="mean_reconstr_cost")
            self.mean_latent_cost = tf.reduce_mean(self.latent_weight * self.latent_loss, name="mean_latent_cost")

        # Combine the costs from each voxel and use a single ADAM optimizer to optimize the mean cost
        # It is also possible to optimize the total cost but this makes it harder to compare with
        # variable numbers of voxels
        #self.mean_cost = tf.reduce_mean(self.cost, name="mean_cost")
        self.mean_cost = tf.add(self.mean_reconstr_cost, self.mean_latent_cost)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.optimize = self.optimizer.minimize(self.mean_cost, global_step=self.global_step)

    def fit_batch(self):
        """
        Train model based on mini-batch of input data.

        :return: Tuple of total cost of mini-batch, latent cost and reconstruction cost
        """
        _, cost, latent, reconstr = self.evaluate(self.optimize, self.cost, self.latent_loss, self.reconstr_loss)
        return cost, latent, reconstr

    def evaluate(self, *tensors):
        """
        Evaluate tensor values

        :param tensors: Sequence of tensors or names of tensors
        :return: If single tensor requested, it's value as Numpy array. Otherwise tuple of Numpy arrays
        """
        actual_tensors = []
        for t in tensors:
            if isinstance(t, six.string_types):
                t = self.sess.graph.get_tensor_by_name("%s:0" % t)
            actual_tensors.append(t)

        out = self.sess.run(actual_tensors, feed_dict=self.feed_dict)
        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)

    def state(self):
        """
        Get the current state of the optimization.

        This can be used to restart from a previous state if a numerical error occurs
        """
        return self.evaluate(self.post.state())
        
    def set_state(self, state):
        """
        Set the state of the optimization

        :param state: State as returned by the ``state()`` method
        """
        self.evaluate(self.post.set_state(state))
        
    def train(self, tpts, data,
              batch_size=None, sequential_batches=False,
              epochs=100, fit_only_epochs=0, display_step=1,
              learning_rate=0.1, lr_decay_rate=1.0,
              sample_size=None, ss_increase_factor=1.0,
              revert_post_trials=50, revert_post_final=True,
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
        :param sequential_batches: If True, form batches from consecutive time points rather than strides
        :param epochs: Number of training epochs
        :param fit_only_epochs: If specified, this number of epochs will be restricted to fitting only
                                and ignore prior information. In practice this means only the
                                reconstruction loss is considered not the latent cost
        :param display_step: How many steps to execute for each display line
        :param learning_rate: Initial learning rate
        :param lr_decay_rate: When adjusting the learning rate, the factor to reduce it by
        :param sample_size: Number of samples to use when estimating expectations over the posterior
        :param ss_increase_factor: Factor to increase the sample size by over the epochs
        :param revert_post_trials: How many epoch to continue for without an improvement in the mean cost before
                                   reverting the posterior to the previous best parameters
        :param revert_post_final: If True, revert to the state giving the best cost achieved after the final epoch
        """
        # Expect tpts to have a dimension for voxelwise variation even if it is the same for all voxels
        if tpts.ndim == 1:
            tpts = tpts.reshape(1, -1)

        # Determine number of voxels and timepoints and check consistent
        n_voxels, n_timepoints = tuple(data.shape)
        n_nodes = self.data_model.n_nodes
        if tpts.shape[0] > 1 and tpts.shape[0] != n_nodes:
            raise ValueError("Time points has %i nodes, but data has %i" % (tpts.shape[0], n_nodes))
        if tpts.shape[1] != n_timepoints:
            raise ValueError("Time points has length %i, but data has %i volumes" % (tpts.shape[1], n_timepoints))

        # Determine number of batches and sample size
        if batch_size is None:
            batch_size = n_timepoints
        n_batches = int(np.ceil(float(n_timepoints) / batch_size))
        if sample_size is None:
            sample_size = batch_size

        # Cost and parameter histories, mean and voxelwise
        training_history = {
            "mean_cost" : np.zeros([epochs+1]),
            "voxel_cost" : np.zeros([n_voxels, epochs+1]),
            "mean_params" : np.zeros([epochs+1, self._nparams]),
            "voxel_params" : np.zeros([n_nodes, epochs+1, self._nparams]),
            "ak" : np.zeros([epochs+1]),
            "runtime" : np.zeros([epochs+1]),
        }

        # Training cycle
        self.feed_dict = {
            self.data_full : data,
            self.num_steps : epochs*n_batches,
            self.initial_lr : learning_rate,
            self.lr_decay_rate : lr_decay_rate,
            self.initial_ss : sample_size,
            self.ss_increase_factor : ss_increase_factor,
            self.data_train: data,
            self.tpts_train : tpts,
            self.latent_weight : 1.0,
        }
        self.evaluate(self.init)

        trials, best_cost, best_state = 0, 1e12, None
        latent_weight = 0

        # Each epoch passes through the whole data but it may do this in 'batches' so there may be
        # multiple training iterations per epoch, one for each batch
        self.log.info("Training model...")
        self.log.info(" - Number of training epochs: %i", epochs)
        self.log.info(" - %i voxels of %i time points (processed in %i batches of target size %i)" , n_voxels, n_timepoints, n_batches, batch_size)
        self.log.info(" - Initial learning rate: %.5f (decay rate %.3f)", learning_rate, lr_decay_rate)
        self.log.info(" - Initial sample size: %i (increase factor %.3f)", sample_size, ss_increase_factor)
        if revert_post_trials > 0:
            self.log.info(" - Posterior reversion after %i trials", revert_post_trials)

        initial_means = np.mean(self.evaluate(self.model_means), axis=1)
        initial_vars = np.mean(self.evaluate(self.post.var), axis=0)
        initial_cost = np.mean(self.evaluate(self.cost))
        initial_latent = np.mean(self.evaluate(self.latent_loss))
        initial_reconstr = np.mean(self.evaluate(self.reconstr_loss))
        start_time = time.time()
        self.log.info(" - Start 0000: mean cost=%f (latent=%f, reconstr=%f) mean params=%s mean_var=%s", 
                      initial_cost, initial_latent, initial_reconstr, initial_means, initial_vars)
        for epoch in range(epochs):
            try:
                err = False
                total_cost = np.zeros([n_voxels])
                total_latent = np.zeros([n_nodes])
                total_reconstr = np.zeros([n_voxels])

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
                        index = i*batch_size
                        if i == n_batches - 1:
                            # Batch size may not be an exact factor of the number of time points
                            # so make the last batch the right size so all of the data is used
                            batch_size += n_timepoints - n_batches * batch_size
                        batch_data = data[:, index:index+batch_size]
                        batch_tpts = tpts[:, index:index+batch_size]
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
                        self.latent_weight : latent_weight,
                    })
                    batch_cost, batch_latent, batch_reconstr = self.fit_batch()
                    total_latent += batch_latent / n_batches
                    total_reconstr += batch_reconstr / n_batches
                    total_cost += batch_cost / n_batches

            except tf.OpError:
                self.log.exception("Numerical error fitting batch")
                err = True

            # Record the cost and parameter values at the end of each epoch.
            params = self.evaluate(self.model_means) # [P, W]
            var = self.evaluate(self.post.var) # [W, P]
            current_lr, current_ss = self.evaluate(self.learning_rate, self.sample_size)
            mean_params = np.mean(params, axis=1)
            median_params = np.median(params, axis=1)
            mean_var = np.mean(var, axis=0)

            mean_total_latent = np.mean(total_latent)
            mean_total_reconst = np.mean(total_reconstr)
            mean_total_cost = mean_total_latent + mean_total_reconst
            median_total_latent = np.median(total_latent)
            median_total_reconst = np.median(total_reconstr)
            median_total_cost = median_total_latent + median_total_reconst # approx
            
            training_history["mean_cost"][epoch] = mean_total_cost
            training_history["voxel_cost"][:, epoch] = total_cost
            training_history["mean_params"][epoch, :] = mean_params
            training_history["voxel_params"][:, epoch, :] = params.transpose()
            try:
                training_history["ak"][epoch] = self.evaluate("ak")
            except:
                pass

            if err or np.isnan(mean_total_cost) or np.any(np.isnan(mean_params)):
                # Numerical errors while processing this epoch. Revert to best saved params if possible
                if best_state is not None:
                    self.set_state(best_state)
                outcome = "Revert - Numerical errors"
            elif mean_total_cost < best_cost:
                # There was an improvement in the mean cost - save the current state of the posterior
                outcome = "Saving"
                best_cost = mean_total_cost
                best_state = self.state()
                trials = 0
            else:
                # The mean cost did not improve. 
                if revert_post_trials > 0:
                    # Continue until it has not improved for revert_post_trials epochs and then revert 
                    trials += 1
                    if trials < revert_post_trials:
                        outcome = "Trial %i" % trials
                    elif best_state is not None:
                        self.set_state(best_state)
                        outcome = "Revert"
                        trials = 0
                    else:
                        outcome = "Continue - No best state"
                        trials = 0
                else:
                    outcome = "Not saving"

            if epoch % display_step == 0:
                state_str = "mean/median cost=%f/%f (latent=%f, reconstr=%f) mean params=%s mean_var=%s lr=%f, ss=%i" % (
                    mean_total_cost, median_total_cost, mean_total_latent, mean_total_reconst, median_params, mean_var, current_lr, current_ss)
                self.log.info(" - Epoch %04d: %s - %s", (epoch+1), state_str, outcome)

            epoch_end_time = time.time()
            training_history["runtime"][epoch] = float(epoch_end_time - start_time)

        if revert_post_final and best_state is not None:
            # At the end of training we revert to the state with best mean cost and write a final history step
            # with these values. Note that the cost may not be as reported earlier as this was based on a
            # mean over the training batches whereas here we recalculate the cost for the whole data set.
            self.log.info("Reverting to best batch-averaged cost")
            self.set_state(best_state)

        self.feed_dict[self.data_train] = data
        self.feed_dict[self.tpts_train] = tpts
        cost = self.evaluate(self.cost) # [W]
        params = self.evaluate(self.model_means) # [P, W]
        
        self.log.info(" - Best batch-averaged cost: %f", best_cost)
        self.log.info(" - Final cost across full data: %f", np.mean(cost))
        self.log.info(" - Final params: %s", np.mean(params, axis=1))

        training_history["mean_cost"][-1] = np.mean(cost)
        training_history["voxel_cost"][:, -1] = cost
        training_history["mean_params"][-1, :] = np.mean(params, axis=1)
        training_history["voxel_params"][:, -1, :] = params.transpose()
        try:
            training_history["ak"][-1] = self.evaluate("ak")
        except:
            pass

        # Return training history
        return training_history
