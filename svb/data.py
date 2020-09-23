"""
SVB - Data model
"""
import math
import collections

import six
import numpy as np
import nibabel as nib
import tensorflow as tf
from scipy import sparse

from .utils import LogBase, TF_DTYPE, NP_DTYPE

class DataModel(LogBase):
    """
    Encapsulates information about the data volume being modelled

    This includes its overall dimensions, mask (if provided), 
    and neighbouring voxel lists
    """

    def __init__(self, data, mask=None, **kwargs):
        LogBase.__init__(self)

        self.nii, self.data_vol = self._get_data(data)
        while self.data_vol.ndim < 4:
            self.data_vol = self.data_vol[np.newaxis, ...]

        self.shape = list(self.data_vol.shape)[:3]
        self.n_tpts = self.data_vol.shape[3]
        self.data_flattened = self.data_vol.reshape(-1, self.n_tpts)

        # If there is a mask load it and use it to mask the data
        if mask is not None:
            mask_nii, self.mask_vol = self._get_data(mask)
            self.mask_flattened = self.mask_vol.flatten()
            self.data_flattened = self.data_flattened[self.mask_flattened > 0]
        else:
            self.mask_vol = np.ones(self.shape)
            self.mask_flattened = self.mask_vol.flatten()

        self.n_unmasked_voxels = self.data_flattened.shape[0]
        print("Data voxels: %i" % self.n_unmasked_voxels)

    @property
    def is_volumetric(self):
        return (type(self) is VolumetricModel)

    # TODO: volumetric only method 
    def nifti_image(self, data):
        """
        :return: A nibabel.Nifti1Image for some, potentially masked, output data
        """
        shape = self.shape
        if data.ndim > 1:
            shape = list(shape) + [data.shape[1]]
        ndata = np.zeros(shape, dtype=np.float)
        ndata[self.mask_vol > 0] = data
        return nib.Nifti1Image(ndata, None, header=self.nii.header)

    def posterior_data(self, mean, cov):
        """
        Get voxelwise data for the full posterior

        We use the Fabber method of saving the upper triangle of the
        covariance matrix concatentated with a column of means and
        an additional 1.0 value to make it square.

        Note that if some of the posterior is factorized or 
        covariance is not being inferred some or all of the covariances
        will be zero.
        """
        if cov.shape[0] != self.n_unmasked_voxels or mean.shape[0] != self.n_unmasked_voxels:
            raise ValueError("Posterior data has %i voxels - inconsistent with data model containing %i unmasked voxels" % (cov.shape[0], self.n_unmasked_voxels))

        num_params = mean.shape[1]
        vols = []
        for row in range(num_params):
            for col in range(row+1):
                vols.append(cov[:, row, col])
        for row in range(num_params):
            vols.append(mean[:, row])
        vols.append(np.ones(mean.shape[0]))
        return np.array(vols).transpose((1, 0))

    def _get_data(self, data):
        if isinstance(data, six.string_types):
            nii = nib.load(data)
            if data.endswith(".nii") or data.endswith(".nii.gz"):
                data_vol = nii.get_data()
            elif data.endswith(".gii"):
                # FIXME
                raise NotImplementedError()
            self.log.info("Loaded data from %s", data)
        else:
            nii = nib.Nifti1Image(data.astype(np.int8), np.identity(4))
            data_vol = data
        return nii, data_vol

    def _get_posterior_data(self, post_data):
        if isinstance(post_data, six.string_types):
            return self._posterior_from_file(post_data)
        elif isinstance(post_data, collections.Sequence):
            return tuple(post_data)
        else:
            raise TypeError("Invalid data type for initial posterior: should be filename or tuple of mean, covariance")

    def _posterior_from_file(self, fname):
        """
        Read a Nifti file containing the posterior saved using --save-post
        and extract the covariance matrix and the means
        
        This can then be used to initialize a new run - note that
        no checking is performed on the validity of the data in the MVN
        other than it is the right size.
        """
        post_data = nib.load(fname).get_data()
        if post_data.ndim !=4:
            raise ValueError("Posterior input file '%s' is not 4D" % fname)
        if list(post_data.shape[:3]) != list(self.shape):
            raise ValueError("Posterior input file '%s' has shape %s - inconsistent with mask shape %s" % (fname, post_data.shape[:3], self.shape))

        post_data = post_data[self.mask_vol > 0]
        nvols = post_data.shape[1]
        self.log.info("Posterior image contains %i volumes" % nvols)

        n_params = int((math.sqrt(1+8*float(nvols)) - 3) / 2)
        nvols_recov = (n_params+1)*(n_params+2) / 2
        if nvols != nvols_recov:
            raise ValueError("Posterior input file '%s' has %i volumes - not consistent with upper triangle of square matrix" % (fname, nvols))
        self.log.info("Posterior image contains %i parameters", n_params)
        
        cov = np.zeros((self.n_unmasked_voxels, n_params, n_params), dtype=NP_DTYPE)
        mean = np.zeros((self.n_unmasked_voxels, n_params), dtype=NP_DTYPE)
        vol_idx = 0
        for row in range(n_params):
            for col in range(row+1):
                cov[:, row, col] = post_data[:, vol_idx]
                cov[:, col, row] = post_data[:, vol_idx]
                vol_idx += 1
        for row in range(n_params):
            mean[:, row] = post_data[:, vol_idx]
            vol_idx += 1
        if not np.all(post_data[:, vol_idx] == 1):
            raise ValueError("Posterior input file '%s' - last volume does not contain 1.0", fname)

        self.log.info("Posterior mean shape: %s, cov shape: %s", mean.shape, cov.shape)
        return mean, cov


class VolumetricModel(DataModel):

    def __init__(self, data, mask=None, **kwargs):
        super().__init__(data, mask=mask, **kwargs)

        # By default parameter space is same as data space
        self.n2v_coo = None
        self.n_nodes = self.n_unmasked_voxels

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def _calc_laplacian(self):
        """
        Laplacian matrix. Note the sign convention is negatives
        on the diagonal, and positive values off diagonal. 
        """

        # Set the laplacian here 
        lap = self.adj_matrix.todok(copy=True)
        lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
        assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
        self.laplacian = lap.tocoo()


    # TODO: subclass this for surface
    def nodes_to_voxels_ts(self, tensor, vertex_axis=0):
        """
        Map parameter vertex-based data to data voxels

        This is for the use case where data is defined in a different space to the
        parameter estimation space. For example we may be estimating parameters on
        a surface mesh, but using volumetric data to train this model.

        :param tensor: TensorFlow tensor of which one axis represents indexing over
                       parameter nodes
        :param vertex_axis: Index of axis of tensor which corresponds to parameter nodes

        :return: TensorFlow tensor with parameter vertex axis replaced by data voxel
                 axis and other tensor entries transformed appropriately to the 
                 data space
        """

        return tensor

            
    # TODO: subclass this for surface
    def nodes_to_voxels(self, tensor, vertex_axis=0):
        """
        Map parameter vertex-based data to data voxels

        This is for the use case where data is defined in a different space to the
        parameter estimation space. For example we may be estimating parameters on
        a surface mesh, but using volumetric data to train this model.

        :param tensor: TensorFlow tensor of which one axis represents indexing over
                       parameter nodes
        :param vertex_axis: Index of axis of tensor which corresponds to parameter nodes

        :return: TensorFlow tensor with parameter vertex axis replaced by data voxel
                 axis and other tensor entries transformed appropriately to the 
                 data space
        """

        return tensor


    def _calc_adjacency_matrix(self):
        """
        Generate adjacency matrix for voxel nearest neighbours.
        Note the result will be a square sparse COO matrix of size 
        (n_unmasked voxels), indexed according to voxels in the mask
        (so index 0 refers to the first un-masked voxel). 
        
        These are required for spatial priors and in practice do not
        take long to calculate so we provide them as a matter of course
        """
        def add_if_unmasked(x, y, z, masked_indices, nns):
            # Check that potential neighbour is not masked and if so
            # add it to the list of nearest neighbours
            idx  = masked_indices[x, y, z]
            if idx >= 0:
                nns.append(idx)

        # Generate a Numpy array which contains -1 for voxels which
        # are not in the mask, and for those which are contains the
        # voxel index, starting at 0 and ordered in row-major ordering
        # Note that the indices are for unmasked voxels only so 0 is
        # the index of the first unmasked voxel, 1 the second, etc.
        # Note that Numpy uses (by default) C-style row-major ordering
        # for voxel indices so the the Z co-ordinate varies fastest
        masked_indices = np.full(self.shape, -1, dtype=np.int)
        nx, ny, nz = tuple(self.shape)
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask_vol[x, y, z] > 0:
                        masked_indices[x, y, z] = voxel_idx
                        voxel_idx += 1

        # Now generate the nearest neighbour lists.
        voxel_nns = []
        indices_nn = []
        voxel_idx = 0
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    if self.mask_vol[x, y, z] > 0:
                        nns = []
                        if x > 0: add_if_unmasked(x-1, y, z, masked_indices, nns)
                        if x < nx-1: add_if_unmasked(x+1, y, z, masked_indices, nns)
                        if y > 0: add_if_unmasked(x, y-1, z, masked_indices, nns)
                        if y < ny-1: add_if_unmasked(x, y+1, z, masked_indices, nns)
                        if z > 0: add_if_unmasked(x, y, z-1, masked_indices, nns)
                        if z < nz-1: add_if_unmasked(x, y, z+1, masked_indices, nns)
                        voxel_nns.append(nns)
                        # For TensorFlow sparse tensor
                        for nn in nns:
                            indices_nn.append([voxel_idx, nn])
                        voxel_idx += 1

        self.adj_matrix = sparse.coo_matrix(
            (np.ones(len(indices_nn)), (np.array(indices_nn).T)), 
            shape=2*[self.n_unmasked_voxels], 
            dtype=NP_DTYPE
        )

        assert not (self.adj_matrix.tocsr()[np.diag_indices(self.n_unmasked_voxels)] != 0).max()

class SurfaceModel(DataModel):

    def __init__(self, data, surfaces, mask=None, **kwargs):
        super().__init__(data, mask=mask, **kwargs)

        self.surfaces = surfaces['LMS']

        # See if we have a vertex-to-voxel linear mapping
        self.n2v_coo = sparse.coo_matrix(kwargs["n2v"])

        if len(self.n2v_coo.shape) != 2:
            raise ValueError("Vertex-to-voxel mapping must be a matrix")
        if self.n2v_coo.shape[0] != self.n_unmasked_voxels:
            raise ValueError("Vertex-to-voxel matrix - number of columns must match number of unmasked voxels")
        self.n_nodes = self.n2v_coo.shape[1]
        self.n2v_coo.data = self.n2v_coo.data.astype(NP_DTYPE)

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self._calc_adjacency_matrix()
        self._calc_laplacian()

    def _calc_laplacian(self):
        """
        Laplacian matrix. Note the sign convention is negatives
        on the diagonal, and positive values off diagonal. 
        """
        
        # Set the laplacian here 
        lap = self.adj_matrix.todok(copy=True)
        lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
        assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
        self.laplacian = lap.tocoo()


    def _calc_adjacency_matrix(self):
        
        surf = self.surfaces
        adj = sparse.dok_matrix(2*(surf.points.shape[0],), dtype=NP_DTYPE)
        for pidx in range(surf.points.shape[0]):
            touched = (surf.tris == pidx).any(1)
            neighbours = np.unique(surf.tris[touched])
            adj[pidx,neighbours] = 1 

        adj[np.diag_indices(surf.points.shape[0])] = 0
        assert not (adj[np.diag_indices(adj.shape[0])] != 0).nnz
        self.adj_matrix = adj.tocoo()


    def nodes_to_voxels(self, tensor, *unused): 

        n2v_tensor = tf.SparseTensor(
            indices=np.array([self.n2v_coo.row, self.n2v_coo.col]).T,
            values=self.n2v_coo.data.astype(NP_DTYPE), 
            dense_shape=self.n2v_coo.shape
        )
        return tf.sparse.sparse_dense_matmul(n2v_tensor, tensor)

    def nodes_to_voxels_ts(self, tensor, *unused):

        n2v_tensor = tf.SparseTensor(
            indices=np.array([self.n2v_coo.row, self.n2v_coo.col]).T,
            values=self.n2v_coo.data.astype(NP_DTYPE), 
            dense_shape=self.n2v_coo.shape
        )
 
        def sparse_mul(dense):
            return tf.sparse.sparse_dense_matmul(n2v_tensor, dense)

        assert len(tensor.shape) == 3, 'not a 3D tensor'
        result = tf.map_fn(
            sparse_mul, tf.transpose(tensor, [2, 0, 1])
            )
        return tf.transpose(result, [1, 2, 0])

