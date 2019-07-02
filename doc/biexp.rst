Tests using Biexponential model
===============================

The biexponential model outputs a sum of exponentials:

.. math::
    M(t) = A_1 \exp{-R_1 t} + A_2 \exp{-R_2 t}

The model parameters are the amplitudes :math:`A_1`, :math:`A_2`
and the decay rates :math:`R_1` and :math:`R_2`.

Although the model is straightforward it can be challenging as
an inference problem as the effect of the two decay rates on the
output is subtle and nonlinear.

Test data
---------

For testing purposes we define the ground truth parameters as:

 - :math:`A_1=10`
 - :math:`A_2=10`
 - :math:`R_1=1`
 - :math:`R_2=10`

The variables within the test data are:

 - The level of noise present. For this test data we use Gaussian
   noise with a standard deviation of 1.0.
 - The number of time points generated. We generate data sets with
   10, 20, 50 and 100 time points (in each case the value of :math:`t`
   ranges from 0 to 5 so only the data resolution changes in each case)

An example timeseries with these parameters is show below (100 time points,
ground truth overlaid on noisy data):

.. image:: images/sample_timeseries.png
    :alt: Sample timeseries

1000 timeseries instances were generated and used for each test

Test variables
--------------

The following variables were investigated

 - The learning rate
 - The batch size (NB this cannot exceed the number of time points)
 - The prior distribution of the parameter
 - The initial posterior distribution of the parameters
 - The use of the numerical (sample-based) calculation of the KL
   divergence on the posterior vs the analytic solution (possible 
   in this case only because both prior and posterior are represented
   by a multivariate normal distribution).

We investigate convergence by calculating the mean of the cost function
across all test instances by epoch. Note that this measure is not directly 
comparable when different priors are used as the closeness of the 
posterior to the prior is part of the cost calculation.

We also consider typical speed of convergence, defined for each voxel as 
the epoch at which it first came within 5% of its best cost. This 
definition is only useful when convergence was eventually achieved.

Effect of learning rate
-----------------------

The learning rate determines the size of optimization steps made by the
gradient optimizer and can be a difficult variable to select. Too high
and the optimizer may repeatedly overshoot the minima and never actually
converge, too low and convergence may simply be too slow. In many machine
learning problems the learning rate is determined by trial and error however
in our case we do not have this luxury as we need to be able to converge
the model fitting on any unseen data without user intervention.

The convergence of the mean cost is shown below by learning rate and 
number of time points.

.. image:: images/conv_lr.png
    :alt: Convergence by learning rate

Although the picture is rather messy some observations can be made:

 - High learning rates are unstable and do not achieve the best cost
   across the data sets
 - Very low learning rates converge too slowly to be useful
 - Even some learning rates which appear to show good smooth convergence
   do not achieve the minimum cost (e.g. LR=0.25, the greeen line on these
   plots)
 - The best learning rates in this case are in the region 0.1 to 0.05 
   which give reliable and generally fast convergence. We will use
   a learning rate of 0.1 in subsequent tests where a single learning
   rate is used.

Effect of batch size and learning rate on best cost achieved
------------------------------------------------------------

Optimization of the cost function proceeds by 'epochs' which consists
of a single pass through all of the data. Batch processing consists
of dividing the data into smaller batches and performing multiple
iterations of the optimization - one for each batch - during an epoch.
Processing the data in batch is a commonly used method to accelerate
convergence and works because updates to the parameters occurs multiple
times during each epoch. The optimization steps are 'noisier' because
they are based on less training samples and this helps to avoid 
converging onto local minima. Of course if the batch size is too small
the optimization may become so noisy that convergence does not occur
at all.

This plot compares the final cost value achieved with variation in
batch size and learning rate:

.. image:: images/best_cost_lr.png
    :alt: Best cost achieved by batch size and learning rate

These results confirm the use of learning rates between 0.1 and 0.05
as optimal across batch sizes. In general large batch sizes can tolerate
higher learning rates whereas small batch sizes can be used with lower learning
rates. This is in line with expectations since high learning rates and 
low batch sizes both imply a 'noisier' optimization and both excessively
high or low noise in the optimization can be problematic.

The optimal batch size in this data was 10-15 which gives the
'flattest' curve, i.e. least affected by variation in the learning rate.
Where batch size is fixed in subsequent tests we use a value of 10.

Effect of prior and initial posterior
-------------------------------------

The following combinations of prior and posterior were used. An inforative
prior was set with a mean equal to the true parameter value and a standard
deviation of 2.0. Non-informative priors were set with a mean of 1 and a
standard deviation of 1e6 for all parameters.

Non-informative initial posteriors were set equal to the non-informative
prior. Informative posteriors were set with a standard deviation of 2.0
and a mean which either matched or did not match the true parameter value.
In addition, an option in the model enabled the posterior amplitude 
parameters to be initialised from the data.

+------------+----------------------------------------------------------------------+
|Code        |Description                                                           |
+------------+----------------------------------------------------------------------+
|i_i         |Informative prior, informative posterior initialised with mean values |
|            |equal to 1.0 for all parameters                                       |
+------------+----------------------------------------------------------------------+
|i_i_init    |Informative prior, informative posterior initialised with true values |
|            |of the decay rates and with amplitude initialised from the data       |
+------------+----------------------------------------------------------------------+
|i_i_true    |Informative prior, informative posterior initialised with true values |
+------------+----------------------------------------------------------------------+
|i_i_wrong   |Informative prior, informative posterior initialised with mean values |
|            |of 1.0 for the decay rate and 100.0 for the amplitudes (i.e. very far |
|            |from the true values)                                                 |
+------------+----------------------------------------------------------------------+
|i_ni        |Informative prior, non-informative posterior                          |
+------------+----------------------------------------------------------------------+
|i_ni_init   |Informative prior, non-informative posterior with amplitude           |
|            |initialised from the data                                             |
+------------+----------------------------------------------------------------------+
|ni_i        |Non-informative prior, informative posterior initialised with mean    |
|            |values equal to 1.0 for all parameters                                |
+------------+----------------------------------------------------------------------+
|ni_i_init   |Non-informative prior, informative posterior initialised with true    |
|            |values of the decay rates and with amplitude initialised from the data|
+------------+----------------------------------------------------------------------+
|ni_i_true   |Non-informative prior, informative posterior initialised with true    |
|            |values                                                                |
+------------+----------------------------------------------------------------------+
|ni_i_wrong  |Non-informative prior, informative posterior initialised with mean    |
|            |values of 1.0 for the decay rate and 100.0 for the amplitudes (i.e.   |
|            |very far from the true values)                                        |
+------------+----------------------------------------------------------------------+
|ni_ni       |Non-informative prior, non-informative posterior                      |
+------------+----------------------------------------------------------------------+
|ni_ni_init  |Non-informative prior, non-informative posterior with amplitude       |
|            |initialised from the data                                             |
+------------+----------------------------------------------------------------------+

.. image:: images/prior_post.png
    :alt: Best cost achieved by prior and posterior combinations

These results show that in terms of absolute convergence there is no significant 
difference between the choice of prior and posterior. Note that the absolute cost
achieved is different between the informative and non-informative priors as 
expected. The exception is the cases where a non-informative initial posterior is
used - these cases do not achieve convergence.

The explanation for this lies in the fact that components of the cost are dependent
on a sample drawn from the posterior. In the case of a non-informative posterior 
samples of realistic sizes cannot be representative and will appear to be essentially
random. This kind of sample will be unable to direct the optimisation towards the true 
posterior and the minimal cost. 

Hence we conclude that the initial posterior must be informative. 
Even if it is a long way from the true solution, an informative posterior will 
still generate a representative sample of itself which can be used to direct the
optimisation to an improved estimate.

The lower set of plots are identical to the upper set apart from using the analytic
solution to the KL divergence between two MVNs. The similarity between these results
and the numerical solution is evidence that the numerical solution should be sufficient
in cases where the prior and posterior cannot be represented as two MVN distributions.

It is reassuring that the cost can converge under a wide variety of prior and posterior
assumptions, however it is useful to consider the effect of these variabels also
on speed of convergence. The results below illustrate this:

.. image:: images/prior_post_conv_speed.png
    :alt: Best cost achieved by prior and posterior combinations

This plot shows the epoch at which each voxel converged (to with 5% of its final values).
The box plot show the median and IQR, while the circles show slow-converging outliers.
For the reasons given above, non-informative posterior test cases were excluded from
this plot.

It is clear that the main impact on convergence speed is the initial posterior. 
Where it is far from the true values (``i_wrong``) convergence is slowest. However
this problem is much less obvious when the priors are informative as this has the
effect of guiding the optimisation to the correct solution.
