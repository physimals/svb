"""
SVB - Data model
"""
import numpy as np
import nibabel as nib

from .utils import LogBase

class DataModel(LogBase):
    """
    Encapsulates information about the data volume being modelled

    This includes its overall dimensions, mask (if provided), 
    and neighbouring voxel lists
    """

    def __init__(self, data_fname, mask_fname=None):
        LogBase.__init__(self)

        self.nii = nib.load(data_fname)
        self.data_vol = self.nii.get_data()
        self.shape = list(self.data_vol.shape)[:3]
        self.n_tpts = self.data_vol.shape[3]
        self.data_flattened = self.data_vol.reshape(-1, self.n_tpts)
        self.log.info("Loaded data from %s", data_fname)

        # If there is a mask load it and use it to mask the data
        if mask_fname:
            mask_nii = nib.load(mask_fname)
            self.mask_vol = mask_nii.get_data()
            self.mask_flattened = self.mask_vol.flatten()
            self.data_flattened = self.data_flattened[self.mask_flattened > 0]
            self.log.info("Loaded mask from %s", mask_fname)
        else:
            self.mask_vol = np.ones(self.shape)
            self.mask_flattened = self.mask_vol.flatten()

        self.n_unmasked_voxels = self.data_flattened.shape[0]
        self._calc_neighbours()
            
    def _calc_neighbours(self):
        """
        Generate nearest neighbour and second nearest neighbour lists
        
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
        voxel_n2s = [[] for voxel in voxel_nns]
        self.indices_n2 = []
        for voxel_idx, nns in enumerate(voxel_nns):
            for nn in nns:
                voxel_n2s[voxel_idx].extend(voxel_nns[nn])
            voxel_n2s[voxel_idx] = [v for v in voxel_n2s[voxel_idx] if v != voxel_idx]
            for n2 in voxel_n2s[voxel_idx]:
                self.indices_n2.append([voxel_idx, n2])

    def nifti_image(self, data):
        """
        :return: A nibabel.Nifti1Image for some, potentially masked, output data
        """
        shape = self.shape
        if data.ndim > 1 and data.shape[1] > 1:
            shape = list(shape) + [data.shape[1]]
        ndata = np.zeros(shape, dtype=np.float)
        ndata[self.mask_vol > 0] = data
        return nib.Nifti1Image(ndata, None, header=self.nii.header)
