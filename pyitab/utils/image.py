import nibabel as ni
import numpy as np
import os

import logging
logger = logging.getLogger(__name__)


def remove_value(image, value, mask=None):
    """This function remove a value from a nibabel image.
    
    Parameters
    ----------
    image : ndarray
        The input image array
    value : float
        The value that should be subtracted from the image
    mask : ndarray, optional
        The mask on which subtraction is performed (the default is None)
    
    Returns
    -------
    image : array
        The array with the value removed.
    """

    
    if not isinstance(mask, np.ndarray):
        mask = np.ones_like(image)
    elif len(mask.shape) != len(image.shape):
        mask = mask[...,np.newaxis]
        
    mask_ = mask / mask
    mask_[np.isnan(mask_)] = 0
        
    image -= (mask_ * value)
    
    return image
    
    
    

def remove_value_nifti(img_fname, value, output_fname, mask_fname=None):
    """This function is used to remove a single value from a 
    nifti file. If a mask is provided, the value is removed only in the mask
    voxels.
    
    Parameters
    ----------
    filename : string
        The filename of the input image
    output_fname : string
        The output filename of the stored image
    value : float
        The value that should be subtracted from the image
    mask : ndarray, optional
        The mask on which subtraction is performed (the default is None)
    
    Returns
    -------
    output_image
        The nifti image
    """

    
    img = ni.load(img_fname)
    
    if mask_fname is not None:
        mask = ni.load(mask_fname).get_fdata()
        
    
    out_img = remove_value(img.get_fdata(), value, mask)
    
    out_img = ni.save(ni.Nifti1Image(out_img, img.affine), output_fname)
    
    return out_img



def remove_mean_nifti(img_fname, output_fname, mask_fname=None):
    """This function is used to remove the average value from a 
    nifti file. 
    If a mask is provided, the average is calculated only using the mask
    nonzero voxels.
    
    Parameters
    ----------
    filename : string
        The filename of the input image
    output_fname : string
        The output filename of the stored image
    mask : :class:`numpy.ndarray`, optional
        The mask on which average is calculated (the default is None)
    
    Returns
    -------
    output_image : :class:`~nibabel.nifti1.Nifti1Image`
        The nifti image
    """

    
    img = ni.load(img_fname)
    
    data = img.get_fdata()
    
    value = data.mean()
    
    mask_data = None
    if mask_fname is not None:
        mask_data = ni.load(mask_fname).get_fdata().squeeze()
        value = data[np.bool_(mask_data)].mean()
       
    out_img = remove_value(data, value, mask_data)
    
    save_map(output_fname, out_img, affine=img.affine)
    
    return out_img



def conjunction_map(a_map, b_map, output_fname, output='mask'):
    """This is a function to perform conjunction operation of two maps (a * b).
    The second map will be considered as a binary image, using nonzero values to
    mask the first one.
    If output is 'mask' the output will be binary image, in the other
    case ('image') the output will be with image_map values in nonzero mask voxels.
    
    Parameters
    ----------
    a_map : string
        The filename of the input image
    b_map : string
        The filename of the input image
    output_fname : string
        The filename of the output image
    output : str, optional | {default='mask', 'image_map'}
        The output type. If 'mask' the output will be a binary image, else
        the values of the conjunction are those of image_map.
    
    """ 
    a_img = ni.load(a_map)
    b_img = ni.load(b_map)
    
    mask_int = np.int_(b_img.get_fdata() != 0)
    
    data = a_img.get_fdata()
    if output == 'mask':
        data /= data
    
    out_img = np.float_(data * mask_int)
    
    save_map(output_fname, out_img, affine=a_img.affine)
    
    return



def afni_converter(afni_fname, output_fname, brick=None):
    """This function converts AFNI *.HEAD / *.BRIK files
    in nii.gz fortmat.
    
    Parameters
    ----------
    afni_fname : string
        Path to the specified HEAD/BRIK file
    output_fname : string
        Path to the output file (N.B. specify the extension)
    brick : int
        The number of the volume to be converted
    
    """

    
    command = "3dTcat -prefix %s %s" % (output_fname, afni_fname)
    if brick is not None:
        command += "[%s]" % (str(brick))
    print(command)
    os.system(command)
    
    img = ni.load(output_fname)
    output = img.get_fdata().squeeze()
    
    save_map(output_fname, output, affine=img.affine)
    
    return


def save_map(filename, map_array, affine=None, return_nifti=True):
    """This function saves an a 3D/4D map in nifti format.
    
    Parameters
    ----------
    filename : string
        The output filename of the stored image
    map_array : ndarray (3D/4D array)
        The array of the map to be stored
    affine : matrix (dim x dim), optional
        Affine transformation of the map
        (the default is None)
    
    """

    if affine is None:
        affine = np.eye(4)
        
    map_zscore = ni.Nifti1Image(map_array, affine)
    ni.save(map_zscore, filename)
    if not return_nifti:
        return None
    return map_zscore
