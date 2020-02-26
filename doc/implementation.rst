Implementation of SVB
=====================

Use of TensorFlow
-----------------

The maximisation of the free energy with respect to the parameters of the posterior
distribution is implemented using the TensorFlow library which provides efficient
calculation of functions of multidimensional arrays. 

TensorFlow uses a compiled-graph model where the elements in a calculation are
represented as nodes of a graph and dependencies are represented as edges. The
graph is set up prior to the actual 
calculation being performed. In this model, nodes are tensors (multidimensional arrays)
or operations which take one or more tensors and output one or more tensor outputs.
For example a matrix multiplication operation would take two tensors and output 
one. 

For operations such as these, TensorFlow is designed to operate efficiently
on large batches of data, so for example a tensor with dimensions ``[1000, 5, 10]``
can be interpreted as 1000 instances of a 5x10 matrix. This can then be matrix-multiplied
by another tensor of dimensions ``[1000, 10, 8]`` resulting in an output with the same
dimensions of ``[1000, 5, 8]`` interpreted as the 1000 output matrix products.
This system is well suited to our problem where we will be performing calculations
on around :math:`10^5` voxels simultaneously. 

Operations can also reduce the 
dimensions of a tensor, for example by calculating the sum or mean across an
axis, or all axes, reshape tensors (transpose axes), extract subsets of tensors
and perform mathematical operations such as elementwise log, square, etc.

Tensors which are not the output of an operation may be set to some constant value,
however they may also be marked as *variables*. When elements of a tensor are marked
as variable they can be changed by an optimization operation in order to minimise
some *cost function*, defined by a set of operations on the tensors in the
graph. In our case since we seek to maximise the free energy we take the cost function 
as its negative. 

Optimizer operations work by calculating the gradient of the cost
function with respect to all variables (using the backpropogation technique) and
performing a gradient-based minimisation algorithm. This minimisation can be
carried out iteratively until the cost function is determined to have converged
to some predefined extent.

Variables and constant values can be freely mixed in a calculation, for example
one can construct a matrix in which the diagonal elements are variable but
the off-diagonal elements are fixed constants.

Main elements in the calculation graph
--------------------------------------

The elements in the calculation graph are implemented as Python classes which,
contain methods to set up the relevant tensors and operations required to
perform their function. We try to avoid assumptions about the nature of these
operations in order to increase flexibility, e.g. we do not constrain how
the posterior generates samples of values.

In TensorFlow, keeping track of the dimensions of tensors is critical to error-free 
execution! In the following sections we use the following symbols to maintain 
clarity:

 - ``V`` is the number of voxels.
 - ``B`` is the number of time points in each voxel during optimization.
 - ``S`` is the number of samples drawn from the posterior in order to approximate
   integrals over the posterior in the stochastic method. 
 - ``P`` is the number of
   parameters in the generative model, including any parameters required to 
   model (including any parameters needed to model the noise component).

.. note::
    For clarity with our intended application to modelling timeseries 
    of volumetric data, we refer to *voxels* and *time points*. However
    more generally voxels can be thought of as independent instances of
    the data being modelled, and time points could be any kind of series
    of data values.

.. note::
    ``B`` may not be the full set of time points available in the data - see
    *Mini-batch processing* below.

Prior
~~~~~

In the stochastic variational Bayes method, we need to be able to 
integrate the expected log PDF of the prior distribution over the posterior,
and this is calculated using a (random, but hopefully representative)
sample of values from the current posterior.

A prior must therefore provide an operation node which takes a sample
tensor of dimension ``[V, P, S]`` and returns a tensor of dimensions ``[V]``
which contains the mean log PDF for each voxel.

.. code-block:: python

    def mean_log_pdf(self, samples):
        """
        :param samples: A tensor of shape [V, P, S] where V is the number
                        of voxels, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples

        :return: A tensor of shape [V] where V is the number of voxels
                 containing the mean log PDF of the parameter samples
                 provided
        """

Posterior
~~~~~~~~~

The posterior must be able to provide samples from itself, i.e. it must provide
an operation node which takes a sample size parameter, ``S`` and returns
a tensor of dimension ``[V, P, S]`` which returns ``S`` samples for each of ``P``
parameters at each of ``V`` voxels. 

.. code-block:: python

    def sample(self, nsamples):
        """
        :param nsamples: Number of samples to return per voxel / parameter

        :return: A tensor of shape [V, P, S]`` where V is the number
                 of voxels, P is the number of parameters in the distribution
                 (possibly 1) and S is the number of samples
        """

In addition the posterior must be able to calculate the expectation integral of the 
log PDF over its own distribution. This is by definition the entropy of the distribution
and therefore in many cases it can be calculated without reference to a sample. However
the sample is available if it is required. This must provide an operation which
returns a tensor of dimension ``[V]`` containing the entropy at each voxel.

.. code-block:: python

    def entropy(self, samples):
        """
        :param samples: A tensor of shape [V, P, S] where V is the number
                        of voxels, P is the number of parameters in the prior
                        (possibly 1) and S is the number of samples.
                        This parameter may or may not be used in the calculation.

        :return Tensor of shape [V] containing voxelwise distribution entropy
        """

Generative model
~~~~~~~~~~~~~~~~

The job of the model is to provide a predicted set of data points given a set of
parameters. However in the stochastic method it must provide a full prediction
for each time point in the input data for each sample of parameter values 
derived from the posterior.

Hence we require an operation which takes a tensor of dimension ``[P x V x S x 1]`` 
containing the values of the parameters for each sample at each voxel and a tensor
of dimension ``[V, 1, B]`` of time points at each voxel and outputs a 
tensor of dimension ``[V, S, B]`` containing the model prediction at each time point 
for each sample at each voxel.

.. code-block:: python

    def evaluate(self, params, t):
        """
        Evaluate the model

        :param t: Time values to evaluate the model at, supplied as a tensor of shape 
                  [1x1xB] (if time values at each voxel are identical) or [Vx1xB]
                  otherwise.
        :param params Sequence of parameter values arrays, one for each parameter.
                      Each array is VxSx1 tensor where V is the number of voxels and
                      S is the number of samples per parameter. This
                      may be supplied as a PxVxSx1 tensor where P is the number of
                      parameters.

        :return: [VxSxB] tensor containing model output at the specified time values
                 for each voxel, and each sample (set of parameter values).
        """

.. note::
    The dimensions of the sampled input parameter values are transposed from those returned
    by the posterior. This is because it is more convenient for the model to have the
    parameter index first so individual parameter tensors can easily be extracted 
    by indexing. This is helpful as different parameters typically play different roles
    in the model.

.. note::
    The dimension of size 1 in the input parameter values is designed to align with 
    the last dimension in the time points tensor (of size ``B``) to allow the
    parameter values to be broadcast across all time points. Similarly the dimension
    of size 1 in the time points tensor allows the same set of time points to be 
    broadcast across the sample dimension ``S``.

.. note::
    The time points may be identical at all voxels, in which case a time point tensor
    of shape ``[1, 1, B]`` may be provided instead. This can typically be handled automatically
    by broadcasting.

Noise model
~~~~~~~~~~~

The noise model is required to calculate the mean log likelihood of the data over the 
sampled values, given the prediction returned by the model. It must define an operation which takes the
actual data tensor with dimensions ``[V, B]``, the model prediction tensor ``[V, S, B]`` and the
sampled values of the noise parameter ``[V, S]``. The operation must return a voxelwise
mean log likelihood tensor with dimensions ``[V]``.

.. code-block:: python

    def log_likelihood(self, data, pred, noise, nt):
        """
        Calculate the log-likelihood of the data

        :param data: Tensor of shape [V, B]
        :param pred: Model prediction tensor with shape [V, S, B]
        :param noise: Noise parameter samples tensor with shape [V, S]
        :return: Tensor of shape[V] containing mean log likelihood of the 
                 data at each voxel with respect to the noise parameters
        """

.. note::
    Currently we are assuming a single noise parameter. This will be relaxes in future
    and the input noise parameter tensor will have dimension ``[V, S, Pn]`` where ``Pn`` is
    the number of noise parameters
    
Cost function
~~~~~~~~~~~~~

The total loss is defined by summing two loss tensors. The *Reconstruction loss* is the
negative of the mean log likelihood returned by the noise model. This is essentially
a measure of how well the model prediction fits the data. The *Latent loss* is the
posterior distribution entropy minus the mean log PDF of the prior and penalises
large deviations from the prior values of the parameters. Each of these is 
defined by operations on the tensors returned by the prior, posterior and noise models
and has dimension ``[V]``.

.. note::
    If both prior and posterior are multivariate Gaussian distributions an analytic
    expression for the latent loss is available which does not require the use
    of a sample. In this case we use this instead of the calculation described above,
    and an additional operation is defined in the MVN posterior for this.

The final cost function is then defined as a mean over voxels of the loss tensor, i.e.
a scalar. This is to ensure that the optimizer has a single value to optimize the
parameters over.

Optimization strategy
---------------------

Optimization is performed using the ``AdamOptimizer``, a gradient based optimization
algorithm that seeks to minimise the given cost function. The key parameter in configuring
the optimizer is the *learning rate* which determines the size of the step in parameter
space that the optimizer takes in order to reduce the cost function. High learning
rates move further and may therefore reduce the cost function more quickly, however
they may also 'overshoot' the actual minimum and fail to converge, or find a local minimum
instead. Low learning rates by contrast move more cautiously towards the minimum however
the resulting convergence may be too slow to be useful.

Unfortunately there is no obvious way to select the optimal learning rate for a given
problem. Typically in machine learning applications a process of trial and error is
involved, with learning curves used as a way to assess the convergence. This is 
not too problematic as the training is often a one-off or occasional step with the
trained model then re-used for multiple applications. In our case, however, we need
to train a model for each application (data set) we process and the ability to select
a suitable learning rate is critical. For this reason we will need to devote some
effort to identifying how to select this parameter for the kind of data we face.

Optimization is divided into *Epochs*, each of which involves the entire data set being
processed and the parameters and cost function updated. This can be done in a single
iteration of the optimizer, passing all the data in, however it is also possible to use a 
*mini-batch* method which can offer some advantages.

Mini-batch training
~~~~~~~~~~~~~~~~~~~

In mini-batch training, the data set is divided into chunks and an optimization step
is performed for each chunk. When all chunks have been processed an epoch is complete
and we start the next epoch with the first chunk again.

There are two main potential advantages to mini-batch training:

1. *Efficiency* - the information contained in the data set does not scale linearly
   with the number of points included, whereas the computational effort often does. 
   Processing half of the data may take half as much time and yet yield an optimization
   iteration nearly as effective as processing the full data. An epoch is then
   be comprised of two optimization iterations rather then one. which should mean
   faster convergence by epochs.
2. *Increasing noise* - One danger of gradient based optimization is local minima.
   A way to reduce the likelihood of the optimization getting stuck in one is to
   introduce an element of noise to the gradients so the optimization will explore
   a wider range of parameter space during the minimization. Smaller batches of
   data will give noisier gradients and may help alleviate this problem to some
   extent.

While this is persuasive it is important to recognize that assumptions about the
efficiency of an optimization must be tested in practice. Typical machine learning
applications often have extremely large numbers of training examples and often use
mini-batch sizes of 20-50. In our case the number of time points in real data
rarely exceeds 100 so it may be the case that mini-batch training is only 
useful for larger data sets.

A mini-batch can be extracted from the data in two main ways, either by dividing
up the data into sequential chunks or by taking strided subsamples through the 
data. The latter seems more approprate when the data forms a continuous timeseries
since we are always using information from across the time series, however for the 
same reason the former method may be preferred when our data consists of repeated 
blocks of measurements of the same timeseries (as is sometimes the case for ASL data).
Our implementation supports both via the ``sequential_batches`` parameter.

One factor that needs to be accounted for when doing mini-batch training is the
scaling of different contributions to the total cost. The latent loss depends 
only on the prior and posterior distributions and not on the size of the training
data, however the reconstruction loss is a sum of log probabilities over the
points in the training data. Correct Bayesian inference only occurs when this
is scaled by :math:`\frac{N_t}{N_b}` where :math:`N_t` is the number of time
points in the full data and :math:`N_b` is the number in the mini-batch, i.e.
the batch is being used to estimate the reconstruction loss for the full
data set.

Learning rate quenching
~~~~~~~~~~~~~~~~~~~~~~~

There is no requirement to keep the learning rate constant throughout the 
optimization. It can, and often is, changed after each epoch or training 
iteration. One simple strategy is to gradually reduce ('quench') the learning rate, starting
off with a high value that quickly explores the parameter space, and reducing it 
to home in on the minimum with high accuracy. Currently we have a very simple
implementation of this idea using the following parameters:

 - ``max_trials`` If this number of epochs passes without the cost function
   improving over the previous best, the learning rate will be reduced
 - ``quench_rate`` - a factor to reduce the learning rate by (e.g. 0.5 means the
   learning rate will be halved)
 - ``min_learning_rate`` - The learning rate will never be reduced 
   lower than this value

This scheme gives us some freedom to start with relatively high learning rates
and reduce them if they are not getting us anywhere. We also adopt the same
strategy where a numerical error is detected. Often this occurs when parameters
stray out of 'reasonable' ranges, suggesting an excessively large optimization
step. In this case we reset to the previous best cost state and reduce the
learning rate by ``quench_rate`` and continue.

It is worth noting that this is far from being the only strategy for modifying
learning rates during training, not is it an agreed best practice! Other
ideas include:

 - Starting with a low learning rate and *increasing* it until the
   cost stops decreasing, thus determining an optimal learning rate which 
   is then selected.
 - Cycling the learning rate to explore a varied region of parameter space
   and aid escape from local minima (possibly combined with quenching over time)
 - Increasing the batch size rather than the learning rate to reduce gradient
   noise as convergence is approached.

It remains to be seen if any of these strategies are useful in our application - 
again they are typically the product of machine learning applications which, 
although they resemble our problem in some ways, differ greatly in others so
not all recommended strategies may be useful. 

Voxelwise convergence
~~~~~~~~~~~~~~~~~~~~~

Our implementation seeks to minimise the mean cost over all voxels, however it
is clear in practice that some voxels converge more rapidly than others. If
we can identify 'converged' voxels and exclude them from the calculation in
subsequent epochs we may attain overall convergence faster (or alternatively
be able to use larger numbers of epochs to ensure we converge 'difficult' 
voxels without penalising our runtime too much).

Two ways we might accomplish this are:

 - A voxelwise mask which selects out a subset of the data for cost 
   calculation. This would need to be applied at an early stage in the
   calculation graph in order to save computational time.
 - 'Zeroing' the gradients of converged voxels so they do not contribute
   to the minimisation.

We have not attempted to implement these strategies yet because currently we
want to understand convergence generally and are less concerned with absolute
performance. However this would be useful to investigate as we start to 
apply the method to real examples.

