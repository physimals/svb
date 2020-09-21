"""
Test for prior classes
"""
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
   
import numpy as np

from svb.prior import NormalPrior, FactorisedPrior

def test_normal_prior_ctor():
    """ Test the normal prior constructor """
    with tf.Session() as session:
        nvoxels_in = 34
        means = np.random.normal(5.0, 3.0, [nvoxels_in, 1])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, 1]))
        prior = NormalPrior(means, variances)

        session.run(tf.global_variables_initializer())
        out_mean = session.run(prior.mean)
        assert np.allclose(out_mean, means)
        out_var = session.run(prior.var)
        assert np.allclose(out_var, variances)

def test_normal_prior_mean_log_pdf():
    """ mean_log_pdf for normal prior """
    with tf.Session() as session:
        nvoxels_in = 34
        means = np.random.normal(5.0, 3.0, [nvoxels_in, 1])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, 1]))
        prior = NormalPrior(means, variances)

        session.run(tf.global_variables_initializer())
        samples = np.random.normal(4.3, 2.8, [nvoxels_in, 1, 100])
        mean_log_pdf = session.run(prior.mean_log_pdf(tf.constant(samples, dtype=TF_DTYPE)))
        assert list(mean_log_pdf.shape) == [nvoxels_in]

        # Check against standard result
        zval = np.square(samples - np.reshape(means, [nvoxels_in, 1, 1])) / np.reshape(variances, [nvoxels_in, 1, 1])
        logpdf = np.reshape(-0.5*zval, [nvoxels_in, -1])
        assert np.allclose(mean_log_pdf, np.mean(logpdf, axis=-1), atol=1e-4)

def test_fac_ctor():
    """ Test factorised prior constructor """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 3
        priors = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        name = "TestFactorisedPrior"
        for param in range(nparams_in):
            priors.append(NormalPrior(means[:, param], variances[:, param]))

        prior = FactorisedPrior(priors, name=name)
        assert prior.name == name
        assert not prior.debug
        assert prior.nparams == nparams_in

def test_fac_mean_log_pdf():
    """ Test mean_log_pdf for factorised prior is sum of contributions """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 3
        priors = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        for param in range(nparams_in):
            priors.append(NormalPrior(means[:, param], variances[:, param]))

        prior = FactorisedPrior(priors)

        samples = np.random.normal(4.3, 2.8, [nvoxels_in, nparams_in, 100])
        tf_samples = tf.constant(samples, dtype=TF_DTYPE)

        session.run(tf.global_variables_initializer())
        out_logpdf = session.run(prior.mean_log_pdf(tf_samples))
        assert list(out_logpdf.shape) == [nvoxels_in]

        in_logpdf = np.zeros([nvoxels_in])
        for idx, p in enumerate(priors):
            tf_samples_param = tf.constant(samples[:, [idx], :], dtype=TF_DTYPE)
            param_logpdf = session.run(p.mean_log_pdf(tf_samples_param))
            in_logpdf += param_logpdf
        assert np.allclose(out_logpdf, in_logpdf)
