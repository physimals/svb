"""
SVB - Data model
"""
import math
import collections

import six
import numpy as np
import nibabel as nib
import tensorflow as tf

from .utils import LogBase

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
        if mask:
            mask_nii, self.mask_vol = self._get_data(mask)
            self.mask_flattened = self.mask_vol.flatten()
            self.data_flattened = self.data_flattened[self.mask_flattened > 0]
        else:
            self.mask_vol = np.ones(self.shape)
            self.mask_flattened = self.mask_vol.flatten()

        self.n_unmasked_voxels = self.data_flattened.shape[0]
        print("Data voxels: %i" % self.n_unmasked_voxels)

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
            nii = nib.Nifti1Image(data, np.identity(4))
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
        
        cov = np.zeros((self.n_unmasked_voxels, n_params, n_params), dtype=np.float32)
        mean = np.zeros((self.n_unmasked_voxels, n_params), dtype=np.float32)
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
        self.v2w_data = None
        self.n_nodes = self.n_unmasked_voxels

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self._calc_neighbours()


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


    def _calc_neighbours(self):
        """
        Generate nearest neighbour and second nearest neighbour lists
        
        These are required for spatial priors and in practice do not
        take long to calculate so we provide them as a matter of course

        FIXME this needs to be done in parameter space where that differs 
        from the data space
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
        self.indices_nn = []
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
                            self.indices_nn.append([voxel_idx, nn])
                        voxel_idx += 1

        # Second nearest neighbour lists exclude self but include duplicates
        # TODO: currently we do not use 2nd neighbours in spatial prior
        voxel_n2s = [[] for voxel in voxel_nns]
        self.indices_n2 = []
        for voxel_idx, nns in enumerate(voxel_nns):
            for nn in nns:
                voxel_n2s[voxel_idx].extend(voxel_nns[nn])
            voxel_n2s[voxel_idx] = [v for v in voxel_n2s[voxel_idx] if v != voxel_idx]
            for n2 in voxel_n2s[voxel_idx]:
                self.indices_n2.append([voxel_idx, n2])



        # # See if we have a vertex-to-voxel linear mapping
        # v2w = kwargs.get("v2w", None)
        # if v2w is not None:
        #     self.v2w = None
        #     if isinstance(v2w, six.string_types):
        #         # For the moment the v2w data is stored as a Numpy savez of a SCIPY CSR sparse matrix
        #         self.v2w_data = np.load(v2w)
        #         print("v2w shape=%s (%i indices, %i values)" % (str(self.v2w_data["shape"]), self.v2w_data["indices"].shape[0], self.v2w_data["data"].shape[0]))
        #         #self.v2w = scipy.sparse.csr_matrix((v2w_data["data"], v2w_data["indices"], v2w_data["indptr"]), shape=v2w_data["shape"])
        #         #self.v2w = tf.SparseTensor(indices=v2w_data["indices"], values=v2w_data["data"], dense_shape=v2w_data["shape"])
        #     else:
        #         self.v2w_data = v2w # FIXME assumes dict with indices, data and shape

        #     if len(self.v2w_data["shape"]) != 2:
        #         raise ValueError("Vertex-to-voxel mapping must be a matrix")
        #     if self.v2w_data["shape"][0] != self.n_unmasked_voxels:
        #         raise ValueError("Vertex-to-voxel matrix - number of columns must match number of unmasked voxels")
        #     self.n_nodes = self.v2w_data["shape"][1]
        # else: