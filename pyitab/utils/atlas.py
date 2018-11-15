import numpy as np
import nibabel as ni
import os
import seaborn as sns

from mne.viz import circular_layout


def find_roi_center(img, roi_value):
    """
    This function gives the x,y,z coordinates of a particular ROI using
    the given segmented image and the image level value used to select the ROI.
    
    Parameters
    ----------
    img : nibabel Nifti1Image instance
        The Nifti1Image instance of the segmented image.
    roi_value : int
        The value of the ROI as represented in the segmented image
        
    Returns
    -------
    xyz : tuple
        A triplets representing the xyz coordinate of the selected ROI.
    """
    
    
    affine = img.affine
    
    mask_ = np.int_(img.get_data()) == roi_value
    ijk_coords = np.array(np.nonzero(mask_)).mean(1)
    
    xyz_coords = ijk_coords * affine.diagonal()[:-1] + affine[:-1,-1]
    
    return xyz_coords



def get_aal_coords(fname):
    """
    Function used to obtain coordinates of the ROIs contained in the AAL90 atlas.
    The atlas used is the 2mm nifti version of the atlas.
    
    Returns
    -------
    coords : n x 3 numpy array
        The array containing n xyz coordinates in MNI space, one for each unique value of the atlas
    """
    atlas90 = ni.load(fname)
    coords = [find_roi_center(atlas90, roi_value=i) for i in np.unique(atlas90.get_data())[1:]]
    
    return np.array(coords)



def get_findlab_coords():
    """
    Function used to obtain coordinates of the networks contained in the findlab atlas.
    The atlas used is the 2mm nifti version of the atlas.
    
    Returns
    -------
    coords : n x 3 numpy array
        The array containing n xyz coordinates in MNI space, one for each unique value of the atlas
    """
    roi_list = os.listdir('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/')
    roi_list.sort()
    findlab = [ni.load('/media/robbis/DATA/fmri/templates_fcmri/0_findlab/'+roi) for roi in roi_list]
    f_coords = []
    for img_ in findlab:

        centers = [find_roi_center(img_, roi_value=np.int(i)) for i in np.unique(img_.get_data())[1:]]
        f_coords.append(np.array(centers))
        
    return np.vstack(f_coords)



def get_atlas90_info(background='black'):
    
    """
    Utility function used to load informations about the atlas used
    
    Parameters
    ----------
    
    atlas_name : string | {'atlas90', 'findlab'}
        A string used to understand the atlas information used for plots.
        
    Returns
    -------
    names : list of string
        The list of ROI names.
    
    colors : list of string
        The list of colors used in other functions
    
    index_ : list of int
        How node values should be ordered if the atlas has another order
        (used to separate left/right in the atlas90)
        
    coords : list of tuple (x,y,z)
        Coordinates of the ROI center (used in plot_connectomics)
        
    networks : list of string
        The list of network names.

    """
    

    coords = get_aal_coords('/media/robbis/DATA/fmri/templates_AAL/atlas90_mni_2mm.nii.gz')
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_AAL/atlas90.cod',
                            delimiter='=',
                            dtype=np.str)
    names = roi_list.T[1]
    names_inv = np.array([n[::-1] for n in names])
    index_ = np.argsort(names_inv)
    names_lr = names[index_]
    names = np.array([n.replace('_', ' ') for n in names])

    dict_ = {'L':'#89CC74', 
             'R':'#7A84CC'}
    colors_lr = np.array([dict_[n[:1]] for n in names_inv])    
    colors_lr = sns.husl_palette(2)
    networks = names

    node_angles = circular_layout(names.tolist(),
                                  names[index_].tolist(),
                                  start_pos=90,
                                  group_boundaries=[0, len(names) / 2.+1],
                                  group_sep=0.)
       
        
    return names, colors_lr, index_, coords, networks, node_angles



def get_findlab_info(background='black'):

    coords = get_findlab_coords()
    roi_list = np.loadtxt('/media/robbis/DATA/fmri/templates_fcmri/findlab_rois.txt', 
                    delimiter=',',
                    dtype=np.str)
    networks = roi_list.T[-2]
    names = roi_list.T[2]
    
    dict_ = {   
                'Auditory':'lightgray', 
                'Basal_Ganglia':'lavender',
                #'Basal_Ganglia':'honeydew', 
                'LECN':'tomato',
                'Language':'darkgrey', 
                'Precuneus':'teal',
                'RECN':'lightsalmon', 
                'Sensorimotor':'plum', 
                'Visuospatial':'slateblue', 
                'anterior_Salience':'yellowgreen',
                #'dorsal_DMN':'lightsteelblue',
                'dorsal_DMN':'cadetblue',
                'high_Visual':'khaki', 
                'post_Salience':'mediumseagreen', 
                'prim_Visual':'gold',
                'ventral_DMN':'lightblue'
            }

    if background == 'white':
        dict_['anterior_Salience'] = 'gray'
        dict_['Basal_Ganglia'] = 'black'
        
        
    colors_lr = np.array([dict_[r.T[-2]] for r in roi_list])
    index_ = np.arange(90)

    _, count_ = np.unique(networks, return_counts=True)
    boundaries = np.cumsum(np.hstack(([0], count_)))[:-1]
    node_angles = circular_layout(names.tolist(), 
                                  names.tolist(), 
                                  start_pos=90, 
                                  group_boundaries=boundaries.tolist(),
                                  group_sep=3.5)
    
    
    return names, colors_lr, index_, coords, networks, node_angles


def get_aalmeg_info(background='black'):

    coords = get_aal_coords('/media/robbis/DATA/fmri/templates_AAL/ROI_MNI_V4.nii')

    labels = np.loadtxt("/media/robbis/DATA/fmri/templates_AAL/ROI_MNI_V4.txt", dtype=np.str)
    node_names = labels.T[1][:99]
    node_idx = np.argsort(np.array([node[-1] for node in node_names]))

    node_angles = circular_layout(node_names.tolist(), 
                                  node_names[node_idx].tolist(), 
                                  start_pos=90, 
                                  group_boundaries=[0, len(node_names) / 2.+1],
                                  )

    colors = sns.husl_palette(2)

    networks = node_names.copy()

    return labels, colors, node_idx, coords, networks, node_angles


def get_atlas_info(atlas_name='findlab'):
    """This function is used to obtain information on 
    different atlases.
    
    Parameters
    ----------
    atlas_name : str, optional ('findlab', 'aal_meg', 'atlas90')
        atlas name to get information
    
    Returns
    -------
    List of informations of the atlas:
        labels : names of nodes (n_nodes, string array)
        colors : colors of each node representing some classification
        node_idx : order of nodes for plot purposes
        coords : MNI coordinates of nodes
        networks : name of networks the node belongs to
        node_angles : angles for circular plot
    """


    mapper = {
        'findlab': get_findlab_info,
        'atlas90': get_atlas90_info,
        'aal_meg': get_aalmeg_info,

    }

    return mapper[atlas_name]()