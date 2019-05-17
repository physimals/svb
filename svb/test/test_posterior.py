"""
Tests for posterior classes
"""
import numpy as np
import tensorflow as tf

from svb.posterior import NormalPosterior, FactorisedPosterior, MVNPosterior

def test_normal_onevox():
    """ When creating a one-voxel normal posterior all input parameters are handled correctly """
    with tf.Session() as session:
        means = np.array([3.0])
        variances = np.array([2.0])
        name = "TestPosterior"
        post = NormalPosterior(means, variances, name=name)
        assert post.name == name
        assert not post.debug

        session.run(tf.global_variables_initializer())

        nvoxels = session.run(post.nvoxels)
        assert nvoxels == 1
        out_mean = session.run(post.mean)
        assert np.allclose(out_mean, means)
        out_var = session.run(post.var)
        assert np.allclose(out_var, variances)
        out_std = session.run(post.std)
        assert np.allclose(out_std, np.sqrt(variances))

def test_normal_onevox_sample():
    """ sample() method returns correct shape of data and is consistent with input mean/var """
    with tf.Session() as session:
        means = np.array([3.0])
        variances = np.array([2.0])

        post = NormalPosterior(means, variances)

        session.run(tf.global_variables_initializer())
        sample = session.run(post.sample(100))
        assert list(sample.shape) == [1, 1, 100]
        # Check for silly values
        assert np.all(sample < 100)
        assert np.all(sample > -100)

def test_normal_onevox_entropy():
    """ entropy() method returns correct shape of data and is consistent with standard
        result for normal distribution"""
    with tf.Session() as session:
        means = np.array([3.0])
        variances = np.array([2.0])

        post = NormalPosterior(means, variances)

        session.run(tf.global_variables_initializer())
        entropy = session.run(post.entropy())
        assert list(entropy.shape) == [1]
        assert np.allclose(entropy, -0.5*np.log(variances))

def test_normal_multivox():
    """ test_normal_onevox for more than one voxel """
    with tf.Session() as session:
        nvoxels_in = 34
        means = np.random.normal(5.0, 3.0, [nvoxels_in, 1])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, 1]))

        post = NormalPosterior(means, variances)

        session.run(tf.global_variables_initializer())

        nvoxels_out = session.run(post.nvoxels)
        assert nvoxels_out == nvoxels_in
        out_mean = session.run(post.mean)
        assert np.allclose(out_mean, means)
        out_var = session.run(post.var)
        assert np.allclose(out_var, variances)
        out_std = session.run(post.std)
        assert np.allclose(out_std, np.sqrt(variances))

def test_normal_multivox_sample():
    """ test_normal_onevox_sample for more than one voxel """
    with tf.Session() as session:
        nvoxels_in = 34
        means = np.random.normal(5.0, 3.0, [nvoxels_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in]))

        post = NormalPosterior(means, variances)

        session.run(tf.global_variables_initializer())
        sample = session.run(post.sample(100))
        assert sample.shape[0] == nvoxels_in
        assert sample.shape[1] == 1
        assert sample.shape[2] == 100
        # Check for silly values
        assert np.all(sample < 100)
        assert np.all(sample > -100)

def test_normal_multivox_entropy():
    """ test_normal_onevox_entropy for more than one voxel """
    with tf.Session() as session:
        nvoxels_in = 34
        means = np.random.normal(5.0, 3.0, [nvoxels_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in]))

        post = NormalPosterior(means, variances)

        session.run(tf.global_variables_initializer())
        entropy = session.run(post.entropy())
        assert list(entropy.shape) == [nvoxels_in]
        assert np.allclose(entropy, -0.5*np.log(variances))

def test_fac_multivox():
    """ Test constructor of factorised posterior """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 34
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        name = "TestFactorisedPosterior"
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = FactorisedPosterior(posts, name=name)
        assert post.name == name
        assert not post.debug
        assert post.nparams == nparams_in

        session.run(tf.global_variables_initializer())
        nvoxels_out = session.run(post.nvoxels)
        assert nvoxels_out == nvoxels_in
        out_mean = session.run(post.mean)
        assert np.allclose(out_mean, means)
        out_var = session.run(post.var)
        assert np.allclose(out_var, variances)
        out_std = session.run(post.std)
        assert np.allclose(out_std, np.sqrt(variances))
        out_cov = session.run(post.cov)
        assert list(out_cov.shape) == [nvoxels_in, nparams_in, nparams_in]
        for vox in range(nvoxels_in):
            assert np.allclose(out_cov[vox, :, :], np.diag(variances[vox, :]))

def test_fac_multivox_sample():
    """ Test sampling from factorised posterior """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 34
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        name = "TestFactorisedPosterior"
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = FactorisedPosterior(posts, name=name)

        session.run(tf.global_variables_initializer())
        sample = session.run(post.sample(100))
        assert list(sample.shape) == [nvoxels_in, nparams_in, 100]
        # Check for silly values
        assert np.all(sample < 100)
        assert np.all(sample > -100)

def test_fac_multivox_entropy():
    """ Test entropy from factorised posterior is sum of entropies of param posteriors"""
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 3
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        name = "TestFactorisedPosterior"
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = FactorisedPosterior(posts, name=name)

        session.run(tf.global_variables_initializer())
        out_entropy = session.run(post.entropy())
        assert list(out_entropy.shape) == [nvoxels_in]

        in_entropy = np.zeros([nvoxels_in])
        for p in posts:
            in_entropy += session.run(p.entropy())
        assert np.allclose(out_entropy, in_entropy)

def test_mvn_multivox():
    """ Test constructor of MVN posterior """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 34
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        name = "TestMVNPosterior"
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = MVNPosterior(posts, name=name)
        assert post.name == name
        assert not post.debug
        assert post.nparams == nparams_in

        session.run(tf.global_variables_initializer())
        nvoxels_out = session.run(post.nvoxels)
        assert nvoxels_out == nvoxels_in
        out_mean = session.run(post.mean)
        assert np.allclose(out_mean, means)
        out_var = session.run(post.var)
        assert np.allclose(out_var, variances)
        out_std = session.run(post.std)
        assert np.allclose(out_std, np.sqrt(variances))

        # Note that covariances are zero initially
        out_cov = session.run(post.cov)
        assert list(out_cov.shape) == [nvoxels_in, nparams_in, nparams_in]
        for vox in range(nvoxels_in):
            assert np.allclose(out_cov[vox, :, :], np.diag(variances[vox, :]))
        out_cov_chol = session.run(post.cov_chol)
        assert list(out_cov_chol.shape) == [nvoxels_in, nparams_in, nparams_in]
        out_off_diag_cov_chol = session.run(post.off_diag_cov_chol)
        assert np.count_nonzero(out_off_diag_cov_chol) == 0
        assert list(out_off_diag_cov_chol.shape) == [nvoxels_in, nparams_in, nparams_in]

def test_mvn_assign_chol_covs():
    """ Check that assigning to covariance variables is reflected in covariance matrix"""
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 13
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = MVNPosterior(posts)
        session.run(tf.global_variables_initializer())

        off_diag = np.full([nvoxels_in, nparams_in, nparams_in], 5.0)
        session.run(post.off_diag_vars.assign(off_diag))
        off_diag_vars = session.run(post.off_diag_vars)
        assert np.allclose(off_diag_vars, 5.0)

        # Check covariance has made it into the off-diagonal of the Cholesky decomp
        cov_chol = session.run(post.cov_chol)
        for i in range(nparams_in):
            for j in range(nparams_in):
                if i == j:
                    assert np.allclose(cov_chol[:, i, j], np.sqrt(variances[:, i]))
                elif i > j:
                    assert np.allclose(cov_chol[:, i, j], 5.0)
                elif j > i:
                    assert np.allclose(cov_chol[:, i, j], 0.0)

        # Check Cholesky decomp seems to be working
        cov = np.matmul(np.transpose(cov_chol, axes=[0, 2, 1]), cov_chol)
        out_cov = session.run(post.cov)
        assert np.allclose(cov, out_cov)

        # Check symmetric which should be guaranteed by Cholesky decomp
        assert np.allclose(out_cov, np.transpose(out_cov, axes=[0, 2, 1]))

def test_mvn_sample():
    """ Test sampling from MVN posterior """
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 34
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = MVNPosterior(posts)

        session.run(tf.global_variables_initializer())
        sample = session.run(post.sample(100))
        assert list(sample.shape) == [nvoxels_in, nparams_in, 100]
        # Check for silly values
        assert np.all(sample < 100)
        assert np.all(sample > -100)

def test_mvn_entropy():
    """ Test entropy from MVN posterior matches standard result"""
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 3
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = MVNPosterior(posts)

        session.run(tf.global_variables_initializer())
        out_entropy = session.run(post.entropy())
        assert list(out_entropy.shape) == [nvoxels_in]

        cov = session.run(post.cov)
        in_entropy = -0.5*np.log(np.linalg.det(cov))
        assert np.allclose(out_entropy, in_entropy)

def test_mvn_entropy_covar():
    """ Test entropy from MVN posterior matches standard result with nonzero covariance"""
    with tf.Session() as session:
        nparams_in = 4
        nvoxels_in = 3
        posts = []
        means = np.random.normal(5.0, 3.0, [nvoxels_in, nparams_in])
        variances = np.square(np.random.normal(2.5, 1.6, [nvoxels_in, nparams_in]))
        for param in range(nparams_in):
            posts.append(NormalPosterior(means[:, param], variances[:, param]))

        post = MVNPosterior(posts)

        session.run(tf.global_variables_initializer())

        # Assign non-zero covariances
        off_diag = np.random.normal(3.2, 5.6, [nvoxels_in, nparams_in, nparams_in])
        session.run(post.off_diag_vars.assign(off_diag))

        out_entropy = session.run(post.entropy())
        assert list(out_entropy.shape) == [nvoxels_in]

        cov = session.run(post.cov)
        in_entropy = -0.5*np.log(np.linalg.det(cov))
        assert np.allclose(out_entropy, in_entropy, atol=1e-3)
