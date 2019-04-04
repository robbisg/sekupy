"""
Transformer for computing seeds signals
----------------------------------------

Mask nifti images by spherical volumes for seed-region analyses
"""
import numpy as np
import sklearn
from sklearn import neighbors
from distutils.version import LooseVersion
from nilearn.image.resampling import coord_transform
import os
from scipy.sparse import load_npz, save_npz
import logging
logger = logging.getLogger(__name__)

def _get_affinity(seeds, 
                  coords, 
                  radius, 
                  allow_overlap, 
                  affine, 
                  mask_img=None):
    
    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.           
    mask_coords = list(zip(*coords.T))
    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T

    if (radius is not None and
            LooseVersion(sklearn.__version__) < LooseVersion('0.16')):
        # Fix for scikit learn versions below 0.16. See
        # https://github.com/scikit-learn/scikit-learn/issues/4072
        radius += 1e-6

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue
        A[i, nearest] = True

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(seed)] = True
        except ValueError:
            # seed is not in the mask
            pass

    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')

    return A



def check_proximity(ds, radius):
    
    logger.info("Checking proximity matrix...")
    radius = np.float(radius)
    fname = os.path.join(ds.a.data_path, "proximity_radius_%s_%s.npz" %(str(radius), ds.a.brain_mask))
    
    return os.path.exists(str(fname))


def load_proximity(ds, radius):
    
    logger.info("Loading proximity matrix...")
    radius = np.float(radius)
    fname = os.path.join(ds.a.data_path, "proximity_radius_%s_%s.npz" %(str(radius), ds.a.brain_mask))
    
    A = load_npz(str(fname))
    return A.tolil()


def save_proximity(ds, radius, A):

    logger.info("Saving proximity matrix...")
    radius = np.float(radius)
    fname = os.path.join(ds.a.data_path, "proximity_radius_%s_%s.npz" %(str(radius), ds.a.brain_mask))
    logger.debug(fname)
    save_npz(str(fname), A.tocoo())


