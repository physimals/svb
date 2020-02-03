Tests using Arterial Spin Labelling model
=========================================

This model implements a basic resting-state ASL kinetic model for PASL
and pCASL acquisitions. The model parameters are :math:`f_{tiss}`, the
relative perfusion and :math:`\delta t` the transit time of the 
blood from the labelling plane to the voxel.

Time points are divided into two categories:

*During bolus* is defined as :math:`\delta t < t <= \tau + \delta t`

*Post bolus* is defined as :math:`t > \tau + \delta t`

Here :math:`\tau` is the bolus duration. The model output is zero for pre-bolus 
time points.

The following rate constant is defined:

:math:`\frac{1}{T_{1app}} = \frac{1}{(1 / T_1 + f_{calib} / \lambda)}`

:math:`\lambda` is the tissue/blood partition coefficient of water which we take to 
be 0.9. :math:`f_{calib}` is the calibrated CBF which typically we do not do not have 
access to (since we are inferring relative CBF) so we use a typical value of 0.01 :math:`s^{-1}`.

CASL model
----------

During bolus
~~~~~~~~~~~~

:math:`M(t) = 2 f_{tiss} T_{1app} \exp{(\frac{-\delta t}{T_{1b}})} (1 - \exp{(-\frac{(t - \delta t)}{T_{1app}})})`

Post bolus
~~~~~~~~~~

:math:`M(t) = 2 f_{tiss} T_{1app} \exp{(-\frac{\delta t}{T_{1b}})} \exp{(-\frac{(t - \tau - \delta t)}{T_{1app}})} (1 - \exp{(-\frac{\tau}{T_{1app}})})`

PASL model
----------

:math:`r = \frac{1}{T_{1app}} - \frac{1}{T_{1b}}`

:math:`f = 2\exp{(-\frac{t}{T_{1app}})}`

During bolus
~~~~~~~~~~~~

:math:`M(t) = f_{tiss} \frac{f}{r} (\exp{(rt)} - \exp{(r\delta t)})`

Post bolus
~~~~~~~~~~
    
:math:`M(t) = f_{tiss} \frac{f}{r} (\exp{(r(\delta t + \tau))} - \exp{(r\delta t)})`

The time points in evaluating an ASL model are the :math:`T_i` values, which may be expressed
as the sum of the bolus duration :math:`\tau` and a post-labelling delay time. For 2D acquisitions
they may be further modified by the additional time delay in acquiring each slice.

Test data
---------

The test data used is a pCASL acquisition with :math:`\tau = 1.8s` and six post-labelling
delays of 0.25, 0.5, 0.75, 1.0, 1.25 and 1.5s. The acquisition was 2D with an additional
time delay of 0.0452s per slice. 8 repeats of the full set of PLDs was obtained.

The test data was fitted in two ways. One method was to average over the repeats
and fit the model to the repeat-free data. The other is to fit the model to the whole
data including repeats. Naturally this involves a larger data size and hence a mini-batch
approach to the optimization.

Mean data tests
~~~~~~~~~~~~~~~

For these tests we have only 6 time points and therefore we do not use a mini-batch
approach, instead using a fixed batch size of 6 (all data points).

Convergence by learning rate
''''''''''''''''''''''''''''

The convergence of mean cost by learning rate is shown below:

.. image:: /images/conv_lr_asl.png
    :alt: Convergence by learning rate

The pattern is closely similar to that obtained using a biexpoential model
although the convergence here is generally 'cleaner'. Learning rates between
0.05 and 0.1 attain the lowest cost within the given number of epochs, with 0.1 
converging faster. Higher learning
rates are less stable and do not appear to be likely to converge, while lower
learning rates converge slowly.

The best cost achieved in 500 epochs is shown below, reinforcing the optimum
learning rate range 0.1 - 0.05

.. image:: /images/best_cost_lr_asl.png
    :alt: Best cost achieved in 500 epochs by learning rate

Full data tests
~~~~~~~~~~~~~~~

For these tests we have 8 repeats of the 6 PLDs giving 48 data points. This
raises the possibility of a mini-batch approach. Intuitively the obvious
choice of batch size is 6, arranged so that each optimization iteration
considers one repeat of all 6 PLDs. However we experiment with varying
the batch size to see if there is any actual advantage in this structure.

.. image:: /images/conv_lr_asl_rpts.png
    :alt: Convergence by learning rate

.. image:: /images/best_cost_lr_asl_rpts.png
    :alt: Best cost achieved in 500 epochs by learning rate

The patterns with convergence and batch size are very similar to those
obtained from the biexponential model. In particular there is no visible
effect of aligning the batch size with the ASL repeats. Again we find
a general optimum learning rate of 0.1 - 0.05 associated with a batch 
size around 10, although it is noticable that the best cost achieved
at lower learning rates is a bit better with smaller batch sizes.
