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
from toblerone.utils import is_symmetric, is_nsd, slice_sparse

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
            self.mask_vol = np.ones(self.shape, dtype=np.bool)
            self.mask_flattened = self.mask_vol.flatten()

        self.n_unmasked_voxels = self.data_flattened.shape[0]

    @property
    def is_volumetric(self):
        return isinstance(self, VolumetricModel)

    @property
    def is_pure_surface(self):
        return type(self) is SurfaceModel

    @property
    def is_hybrid(self):
        return type(self) is HybridModel

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

        :param mean: Posterior mean values as Numpy array [W, P]
        :param cov: Posterior covariance as Numpy array [W, P, P]
        :return: MVN structure as Numpy array [W, Q] where Q is the number of upper triangle elements
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
        return np.stack(vols, axis=-1)

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
        return nii, data_vol.astype(NP_DTYPE)

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

    def nodes_to_voxels(self, tensor, edge_scale=True):
        """
        Map parameter node-based data to voxels

        This is for the use case where data is defined in a different space to the
        parameter estimation space. For example we may be estimating parameters on
        a surface mesh, but using volumetric data to train this model. The input
        data is defined in terms of 'voxels' while the parameter estimation is
        defined on 'nodes'.

        :param tensor:      Tensor with axis 0 representing indexing over nodes
        :param edge_scale:  If True, downweight signal in voxels less than 100%
                            brain (eg, perfusion data). If false, simply perform
                            weighted averaging (eg timing data).  

        :return: Tensor with axis 0 representing indexing over voxels
        """
        raise NotImplementedError()

    def nodes_to_voxels_ts(self, tensor, edge_scale=True):
        """
        Map parameter node-based timeseries data to voxels

        See :func:`~svb.data.DataModel.nodes_to_voxels`

        This method works on time series data, i.e. the conversion to voxel space
        is batched over the outermost axis which is assumed to contain multiple
        independent tensors each requiring conversion.

        :param tensor:      3D tensor of which axis 0 represents indexing over
                            parameter nodes, and axis 2 represents a time series
        :param edge_scale:  If True, downweight signal in voxels less than 100%
                            brain (eg, perfusion data). If false, simply perform
                            weighted averaging (eg timing data).  

        :return: 3D tensor with axis 0 representing indexing over voxels
        """
        raise NotImplementedError()

    def voxels_to_nodes(self, tensor, edge_scale=True):
        """
        Map voxel-based data to nodes. Approximate inverse of :func:`~svb.data.DataModel.nodes_to_voxels`

        :param tensor:      Tensor of which axis 0 represents indexing over voxels
        :param edge_scale:  If True, voxels less than 100% brain will be up-scaled 
                            to compenste for 'missing' signal (eg, perfusion);
                            if False, no scaling will be applied to these voxels
                            (eg timing information). 

        :return: Tensor with axis 0 representing indexing over nodes
        """
        raise NotImplementedError()

    def voxels_to_nodes_ts(self, tensor, edge_scale=True):
        """
        Map voxel-based timeseries data to nodes

        See :func:`~svb.data.DataModel.nodes_to_voxels_ts`

        :param tensor:      3D tensor of which axis 0 represents indexing over
                            voxels, and axis 2 represents a time series
        :param edge_scale:  If True, voxels less than 100% brain will be up-scaled 
                            to compenste for 'missing' signal (eg, perfusion);
                            if False, no scaling will be applied to these voxels
                            (eg timing information). 

        :return: 3D tensor with axis 0 representing indexing over nodes
        """
        raise NotImplementedError()

    def uncache_tensors(self):
        """
        Delete any saved references to tensors

        Used when building a new graph using an existing DataModel
        Yes this is a hack. Got a better idea? Let me know.
        """
        pass

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

        self.adj_matrix = _calc_volumetric_adjacency(self.mask_vol, distance_weight=0)
        self.laplacian = _convert_adjacency_to_laplacian(
                            _calc_volumetric_adjacency(self.mask_vol, distance_weight=1))

    def nodes_to_voxels_ts(self, tensor, edge_scale=True):
        return tensor

    def nodes_to_voxels(self, tensor, edge_scale=True):
        return tensor

    def voxels_to_nodes(self, tensor, edge_scale=True):
        return tensor

    def voxels_to_nodes_ts(self, tensor, edge_scale=True):
        return tensor


class SurfaceModel(DataModel):

    def __init__(self, data, projector, mask=None, **kwargs):
        super().__init__(data, mask=mask, **kwargs)

        self.projector = projector 
        s2v = projector.surf2vol_matrix(edge_scale=True).astype(NP_DTYPE)
        s2v_noedge = projector.surf2vol_matrix(edge_scale=False).astype(NP_DTYPE)
        v2s = projector.vol2surf_matrix(edge_scale=True).astype(NP_DTYPE)
        v2s_noedge = projector.vol2surf_matrix(edge_scale=False).astype(NP_DTYPE)
        assert self.mask_flattened.size == s2v.shape[0], 'Mask size does not match projector'

        # Knock out voxels not included in the mask. 
        vox_inds = np.flatnonzero(self.mask_flattened)
        self.n2v_coo = s2v[vox_inds,:].tocoo()
        self.n2v_noedge_coo = s2v_noedge[vox_inds,:].tocoo()
        self.v2n_coo = v2s[:,vox_inds].tocoo()
        self.v2n_noedge_coo = v2s_noedge[:,vox_inds].tocoo()

        if len(self.n2v_coo.shape) != 2:
            raise ValueError("Vertex-to-voxel mapping must be a matrix")
        if self.n2v_coo.shape[0] != self.n_unmasked_voxels:
            raise ValueError("Vertex-to-voxel matrix - number of columns must match number of unmasked voxels")
        self.n_nodes = self.n2v_coo.shape[1]

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self.adj_matrix = projector.adjacency_matrix(distance_weight=0).tocoo().astype(NP_DTYPE)
        self.laplacian = projector.mesh_laplacian(distance_weight=1).tocoo().astype(NP_DTYPE)

    @property
    def n2v_tensor(self):
        if not hasattr(self, "_n2v_tensor"):
            self._n2v_tensor = tf.SparseTensor(
                indices=np.array([self.n2v_coo.row, self.n2v_coo.col]).T,
                values=self.n2v_coo.data,
                dense_shape=self.n2v_coo.shape)
        return self._n2v_tensor

    @property
    def n2v_noedge_tensor(self):
        if not hasattr(self, "_n2v_noedge_tensor"):
            self._n2v_noedge_tensor = tf.SparseTensor(
                indices=np.array([self.n2v_noedge_coo.row, self.n2v_noedge_coo.col]).T,
                values=self.n2v_noedge_coo.data,
                dense_shape=self.n2v_noedge_coo.shape)
        return self._n2v_noedge_tensor

    @property
    def v2n_tensor(self):
        if not hasattr(self, "_v2n_tensor"):
            self._v2n_tensor = tf.SparseTensor(
                indices=np.array([self.v2n_coo.row, self.v2n_coo.col]).T,
                values=self.v2n_coo.data,
                dense_shape=self.v2n_coo.shape)
        return self._v2n_tensor

    @property
    def v2n_noedge_tensor(self):
        if not hasattr(self, "_v2n_noedge_tensor"):
            self._v2n_noedge_tensor = tf.SparseTensor(
                indices=np.array([self.v2n_noedge_coo.row, 
                                  self.v2n_noedge_coo.col]).T,
                values=self.v2n_noedge_coo.data,
                dense_shape=self.v2n_noedge_coo.shape)
        return self._v2n_noedge_tensor

    def nodes_to_voxels(self, tensor, edge_scale=True):
        if edge_scale:
            return tf.sparse.sparse_dense_matmul(self.n2v_tensor, tensor)
        else:
            raise tf.sparse.sparse_dense_matmul(self.n2v_noedge_tensor, tensor)

    def nodes_to_voxels_ts(self, tensor, edge_scale=True):
        assert len(tensor.shape) == 3, 'not a 3D tensor'
        result = tf.map_fn(lambda t: self.nodes_to_voxels(t, edge_scale), 
                                            tf.transpose(tensor, [2, 0, 1]))
        return tf.transpose(result, [1, 2, 0])

    def voxels_to_nodes(self, tensor, edge_scale=True):
        if edge_scale:
            return tf.sparse.sparse_dense_matmul(self.v2n_tensor, tensor)
        else:
            return tf.sparse.sparse_dense_matmul(self.v2n_noedge_tensor, tensor)

    def voxels_to_nodes_ts(self, tensor, edge_scale=True):
        assert len(tensor.shape) == 3, 'not a 3D tensor'
        result = tf.map_fn(lambda t: self.voxels_to_nodes(t, edge_scale), 
                                            tf.transpose(tensor, [2, 0, 1]))
        return tf.transpose(result, [1, 2, 0])

    def uncache_tensors(self):
        for attrname in ["_n2v_tensor", "_n2v_noedge_tensor",
                        "_v2n_tensor", "_v2n_noedge_tensor"]:
            if hasattr(self, attrname):
                delattr(self, attrname)

    
class HybridModel(SurfaceModel):

    def __init__(self, data, projector, mask=None, **kwargs):

        DataModel.__init__(self, data, mask, **kwargs)

        # Process the projector, apply the mask 
        self.projector = projector
        n2v = projector.node2vol_matrix(edge_scale=True).astype(NP_DTYPE)
        n2v_noedge = projector.node2vol_matrix(edge_scale=False).astype(NP_DTYPE)
        v2n = projector.vol2node_matrix(edge_scale=True).astype(NP_DTYPE)
        v2n_noedge = projector.vol2node_matrix(edge_scale=False).astype(NP_DTYPE)

        if not self.mask_flattened.size == n2v.shape[0]: 
            raise ValueError('Mask size does not match projector')
        if not self.mask_flattened.size + projector.n_surf_points == n2v.shape[1]:
            raise ValueError('Mask size does not match projector')

        # Knock out voxels from projection matrices that are not in the mask
        # We need to shift indices to account for the offset caused by having 
        # all surface vertices come first (order is L surf, R surf, volume)
        vox_inds = np.flatnonzero(self.mask_flattened)
        node_inds = np.concatenate((np.arange(projector.n_surf_points), 
                                    projector.n_surf_points + vox_inds))
        self.n2v_coo = slice_sparse(n2v, vox_inds, node_inds).tocoo()
        self.n2v_noedge_coo = slice_sparse(n2v_noedge, vox_inds, node_inds).tocoo()
        self.v2n_coo = slice_sparse(v2n, node_inds, vox_inds).tocoo()
        self.v2n_noedge_coo = slice_sparse(v2n_noedge, node_inds, vox_inds).tocoo()

        if len(self.n2v_coo.shape) != 2:
            raise ValueError("Vertex-to-voxel mapping must be a matrix")
        if self.n2v_coo.shape[0] != self.n_unmasked_voxels:
            raise ValueError("Vertex-to-voxel matrix - number of columns must match number of unmasked voxels")
        self.n_nodes = self.n2v_coo.shape[1]

        if kwargs.get("initial_posterior", None):
            self.post_init = self._get_posterior_data(kwargs["initial_posterior"])
        else:
            self.post_init = None

        self.adj_matrix = self._calc_adjacency_matrix(distance_weight=0)
        self.laplacian = self._calc_laplacian_matrix(distance_weight=1)

    def _calc_adjacency_matrix(self, distance_weight=0, vox_size=np.ones(3)):
        """
        Construct adjacency matrix for surface and volumetric data. This is
        formed by concatenating in block diagonal form the L surface, 
        R surface and volumetric matrices, in that order. 

        :param distance_weight: int > 0, apply inverse distance weighting, 
            default 0 (do not weight, all egdes are unity), whereas positive
            values will weight edges by 1 / d^n, where d is geometric 
            distance between vertices. 
        :param vox_size: array of 3 values, used for distance weighting of 
            anisotropic voxel grids.   

        :return: sparse COO matrix, of square size (n_verts + n_unmasked_voxels)      
        """
        surf_adj = self.projector.adjacency_matrix(distance_weight)
        vol_adj = _calc_volumetric_adjacency(self.mask_vol, distance_weight, vox_size)
        return sparse.block_diag([surf_adj, vol_adj]).astype(NP_DTYPE)

    def _calc_laplacian_matrix(self, distance_weight=1, vox_size=np.ones(3)):
        """
        Construct Laplacian matrix for surface and volumetric data. This is
        formed by concatenating in block diagonal form the L surface, 
        R surface and volumetric matrices, in that order. 

        :param distance_weight: int > 0, apply inverse distance weighting, 
            default 0 (do not weight, all egdes are unity), whereas positive
            values will weight edges by 1 / d^n, where d is geometric 
            distance between vertices. 
        :param vox_size: array of 3 values, used for distance weighting of 
            anisotropic voxel grids.   

        :return: sparse COO matrix, of square size (n_verts + n_unmasked_voxels)      
        """
        surf_lap = self.projector.mesh_laplacian(distance_weight)
        vol_adj = _calc_volumetric_adjacency(self.mask_vol, distance_weight, vox_size)
        vol_lap = _convert_adjacency_to_laplacian(vol_adj)
        return sparse.block_diag([surf_lap, vol_lap]).astype(NP_DTYPE)




def _convert_adjacency_to_laplacian(adj_matrix):
    """
    Convert an adjacency matrix into a Laplacian. This is done by setting
    the diagonal as the negative sum of each row (note the sign convention
    of positive off-diagonal, negative diagonal). 

    :param adj_matrix: square sparse adjacency matrix of any type. 

    :return: sparse COO matrix
    """
    if not adj_matrix.shape[0] == adj_matrix.shape[1]: 
        raise ValueError("Adjacency matrix is not square")

    lap = adj_matrix.todok(copy=True)
    lap[np.diag_indices(lap.shape[0])] = -lap.sum(1).T
    assert lap.sum(1).max() == 0, 'Unweighted Laplacian matrix'
    return lap.tocoo().astype(NP_DTYPE)


def _calc_volumetric_adjacency(mask, distance_weight=1, vox_size=np.ones(3)):
    """
    Generate adjacency matrix for voxel nearest neighbours.
    Note the result will be indexed according to voxels in the mask (so 
    index 0 refers to the first un-masked voxel). Inverse distance weighting 
    can be applied for non-isotropic voxel sizes (1 / d^n). 
    
    :param distance_weight: int > 0, apply inverse distance weighting, 
        default 0 (do not weight, all egdes are unity), whereas positive
        values will weight edges by 1 / d^n, where d is geometric 
        distance between vertices. 
    :param vox_size: array of 3 values, used for distance weighting of 
        anisotropic voxel grids. 

    :return: square sparse COO matrix of size (mask.sum). 
    """

    # TODO: vectorise this: starting from mask flatnonzero, get all neighbours
    # by adding/subtracting strides from indices, then apply fltr on neighbours
    # array at once, then form CSR matrix directly (indptr is the cumsum of the 
    # fltr row sum). 

    # Voxel neighbours ±1 in each dimension from voxel [0,0,0]
    cardinal_neighbours = np.array([
        [-1,0,0],[1,0,0],[0,-1,0],
        [0,1,0],[0,0,-1],[0,0,1]], 
        dtype=np.int32)

    # Distance weighting is 1 / d^n in each dimension (note the tiling here matches)
    # the order of cardinal neighbours above, ie, XX YY ZZ. 
    dist_weights = (1 / np.tile(vox_size, (2,1)).T.flatten()) ** distance_weight

    # Construct matrix in COO (each entry with row and col index). Iterate only 
    # over voxels we know are in the mask, but indexed into the complete grid
    # (so the first voxel in the mask isn't necessarily voxel index 0)
    shape = mask.shape
    flat_mask = np.flatnonzero(mask)
    rows, cols, entries = [], [], []

    # For each voxel, get cardinal neighbours and mask out those not in
    # the grid. Convert to flat indices, select the corresponding dist
    # weights, and append into the matrix. We are constructing one row 
    # at a time. 
    for idx,vijk in zip(flat_mask, 
                np.array(np.unravel_index(flat_mask, shape)).T):
        neighbours = vijk + cardinal_neighbours
        fltr = (neighbours >= 0).all(-1) & (neighbours < shape).all(-1)
        neighbours = neighbours[fltr,:]
        neighbours_inds = np.ravel_multi_index(neighbours.T, shape)
        rows.append(neighbours_inds.size * [idx])
        cols.append(neighbours_inds)
        entries.append(dist_weights[fltr])
    
    size = np.prod(shape)
    entries = np.concatenate(entries)
    cols = np.concatenate(cols).astype(np.int32)
    rows = np.concatenate(rows).astype(np.int32)
    adj_mat = sparse.coo_matrix((entries, (rows, cols)), shape=(size,size))
    
    adj_mat = slice_sparse(adj_mat, flat_mask, flat_mask).tocoo()
    assert is_symmetric(adj_mat), 'Volumetric adjacency not symmetric'
    return adj_mat.tocoo().astype(NP_DTYPE)
