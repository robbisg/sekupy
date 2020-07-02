import numpy as np
import nibabel as ni
import os
import seaborn as sns

from mne.viz import circular_layout

import logging
logger = logging.getLogger(__name__)


currdir = os.path.dirname(os.path.abspath(__file__))
currdir = os.path.abspath(os.path.join(currdir, os.pardir))
atlasdir = os.path.join(currdir, 'io', 'data', 'atlas')


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
    """Function used to obtain coordinates of the ROIs contained in the AAL90 atlas.
    The atlas used is the 2mm nifti version of the atlas.

    Parameters
    ----------
    fname : string
        The path of the atlas to be used.
    
    Returns
    -------
    coords : n x 3 numpy array
        The array containing n xyz coordinates in MNI space, one for each unique value of the atlas
    """

    atlas90 = ni.load(fname)
    coords = [find_roi_center(atlas90, roi_value=i) for i in np.unique(atlas90.get_data())[:]]
    
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
    atlas_dir = os.path.join(atlasdir, 'findlab')
    roi_list = os.listdir(atlas_dir)
    roi_list.sort()
    findlab = [ni.load(os.path.join(atlas_dir, roi)) for roi in roi_list if roi.find("nii") != -1]
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
    
    background : string | {'black', 'white'}
        A string used to build colors for plots.
        
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
    
    atlas_dir =  os.path.join(atlasdir, 'aal')
    coords = get_aal_coords(os.path.join(atlas_dir, "atlas90_mni_2mm.nii.gz"))
    roi_list = np.loadtxt(os.path.join(atlas_dir, "atlas90.cod"),
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
    roi_file =  os.path.join(atlasdir, 'findlab', 'findlab_rois.txt')
    roi_list = np.loadtxt(roi_file, delimiter=',', dtype=np.str)
    networks = roi_list.T[-2]
    names = roi_list.T[2]
    
    dict_ = {   
                'Auditory'          :'lightgray', 
                'Basal_Ganglia'     :'lavender',
                #'Basal_Ganglia'    :'honeydew', 
                'LECN'              :'tomato',
                'Language'          :'darkgrey', 
                'Precuneus'         :'teal',
                'RECN'              :'lightsalmon', 
                'Sensorimotor'      :'plum', 
                'Visuospatial'      :'slateblue', 
                'anterior_Salience' :'yellowgreen',
                #'dorsal_DMN'       :'lightsteelblue',
                'dorsal_DMN'        :'cadetblue',
                'high_Visual'       :'khaki', 
                'post_Salience'     :'mediumseagreen', 
                'prim_Visual'       :'gold',
                'ventral_DMN'       :'lightblue'
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


def get_aalmeg_info(background='black', grouping='LR'):
    """[summary]
    
    Parameters
    ----------
    background : str, optional
        [description], by default 'black'
    grouping : str, optional
        [description], by default 'LR' or 'lobes'
    
    Returns
    -------
    [type]
        [description]
    """

    atlas_fname = os.path.join(atlasdir, 'aal', 'ROI_MNI_V4.nii')
    labels_fname = os.path.join(atlasdir, 'aal', 'ROI_MNI_V4.csv')
    coords = get_aal_coords(atlas_fname)

    labels = np.loadtxt(labels_fname, dtype=np.str, delimiter=',')
    node_names = labels.T[1][:99]

    if grouping == 'LR':
        node_idx = np.argsort(np.array([node[-1] for node in node_names]))
        group_boundaries = [0, len(node_names) / 2.+1]
        colors = sns.husl_palette(2)
        networks = node_names.copy()
    else:
        node_network = labels.T[3][:99]
        node_idx = np.argsort(node_network)
        networks, count = np.unique(node_network, return_counts=True)
        group_boundaries = np.cumsum(np.hstack(([0], count)))[:-1]
        color_network = sns.color_palette("Paired", len(networks)+1)
        colors_ = dict(zip(networks, color_network[1:]))
        colors = [colors_[n] for n in node_network]

    order = node_names[node_idx].tolist()
    node_angles = circular_layout(node_names.tolist(), 
                                  order,
                                  start_pos=90, 
                                  group_boundaries=group_boundaries,
                                  group_sep=3.
                                  )


    return labels, colors, node_idx, coords, networks, node_angles


def get_viviana_info():

    network = ['DAN','VAN','SMN','VIS','AUD','LAN','DMN']
    number = [6, 5, 8, 10, 4, 5, 7]
    color = ['lightgray', 'lavender', 'honeydew', 'tomato', 
              'darkgrey', 'teal', 'lightsalmon']

    labels = ["%s_%02d" % (network[x], i+1) for x in range(len(network)) for i in range(number[x])]
    colors = [color[x] for x in range(len(network)) for i in range(number[x])]

    node_idx = np.arange(len(labels))

    coords = np.random.randint(-20, 20, (3, len(labels)))
    networks = [network[x] for x in range(len(network)) for i in range(number[x])]

    group_boundaries = np.cumsum(np.hstack(([0], number)))[:-1]
    node_angles = circular_layout(labels, 
                                  labels,
                                  start_pos=90, 
                                  group_boundaries=group_boundaries,
                                  group_sep=3.
                                  )
    
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
        'viviana': get_viviana_info,

    }

    return mapper[atlas_name]()


def where_am_i(fname):
    """This function is used to find the ROI name in a mask file using 
    AFNI whereami.
    
    Parameters
    ----------
    fname : string,
        The complete filename of the mask.

    Returns
    -------
    table : list
        The list has a lenght equal to the number of ROIs contained in the file
        Elements of the list:
        (x, y, z): coordinates of ROI's center of mass.
        label: name of the brain area closest to the center of mass
        num: number of voxels included in the ROI
        value: value of the ROI in the input file 

    """

    logger.info("Looking for ROI names in "+fname)
    img = ni.load(fname)
    center_711 = np.array([-70.5, -105, -60.])
    data = img.get_data().squeeze()
    table = []
    for f in np.unique(data)[1:]:
        mask_roi = data == f

        center_mass = np.mean(np.nonzero(mask_roi), axis=1)
        x,y,z = np.rint([3,3,3]*center_mass+center_711)

        command = "whereami %s %s %s -lpi -space TLRC -tab" %(str(x+2.), str(y+2), str(z+2))
        var = os.popen(command).read()
        lines = var.split("\n")
        index = [i for i, l in enumerate(lines) if l[:5] == 'Atlas']
        label1 = lines[index[0]+1]
        label2 = lines[index[0]+2]
        if label1[0] == '*':
            area1 = area2 = "None"
        else:
            area1 = label1.split("\t")[2]
            area2 = label2.split("\t")[2]
        table.append([x, y, z, area1, np.count_nonzero(mask_roi), f])

    return table

def get_rois_names(path):
    import glob
    big_table = {}
    rois = glob.glob(path+"*mask.nii.gz")
    for fname in rois:
        print(fname)
        table = where_am_i(fname)
        key = fname.split('/')[-1].split('.')[0][:-5]
        big_table[key] = table
    
    return big_table