
import matplotlib.pyplot as plt
import numpy as np
from pyitab.utils.atlas import  get_atlas_info
import os
from scipy.stats import zscore
from numpy.ma.core import masked_array
import seaborn as sns
from itertools import cycle
from pyitab.utils.matrix import copy_matrix, array_to_matrix
from mne.viz import circular_layout
from mne.viz.circle import _plot_connectivity_circle_onpick

import logging
logger = logging.getLogger(__name__)


def plot_matrix(matrix, roi_names, networks, threshold=None, **kwargs):
    """
    This function is used to plot connections in square matrix form.
    
    Parameters
    ----------
    
    matrix : numpy array (n x n) float
            The values of connectivity between each of n ROI
            
    roi_names :  list of n string
            The names of each of the n ROI
            
    networks : list of p string
            List of names representing the networks subdivision
            
    threshold : int
            Indicates the value of the most important connections
            
    ticks_type : {'networks', 'roi'}, optional
            Indicates if the tick names should be ROI or networks
            
    ticks_color : list of colors, optional
            The list in matplotlib formats of colors used to
            color the ticks names, this should be in line with
            the ticks_type choice: p colors if we choose 'networks'
            
    facecolor : string, optional
            As in matplotlib it indicates the background color of 
            the plot
            
    
    Returns
    -------
    
    f : matplotlib figure
            The figure just composed.
    
    """ 
    
    
    _plot_cfg = {'ticks_type':'networks',
                 'ticks_color': sns.color_palette("Paired", n_colors=matrix.shape[0]),#get_atlas_info('findlab')[1],
                 'facecolor':'k',
                 'zscore': True
                 }
    
    
    _plot_cfg.update(kwargs)
    
    facecolor_ = _plot_cfg['facecolor']
    if facecolor_ == 'k':
        ax_color = 'white'
    else:
        ax_color = 'k'
        facecolor_ = 'white'
    
    f = plt.figure(figsize=(16., 12.), facecolor=facecolor_, dpi=300)
    #f = plt.figure()
    a = f.add_subplot(111)
    
    max_value = np.max(np.abs(matrix))
    
    # plot gray values
    if _plot_cfg['zscore'] == True:
        z_matrix = (matrix - matrix.mean())/matrix.std()
    else:
        z_matrix = matrix
    
    max_value = np.max(np.abs(z_matrix))

    ax = a.imshow(z_matrix, 
                  interpolation='nearest', 
                  cmap=plt.cm.RdBu_r,
                  #cmap='Greys',
                  alpha=0.5,
                  vmax=max_value,
                  vmin=max_value*-1
                  )
    
    if threshold is not None:
        thresh_matrix = masked_array(z_matrix, (np.abs(z_matrix) < threshold))
        ax = a.imshow(thresh_matrix,
                      interpolation='nearest', 
                      cmap=plt.cm.bwr,
                      #cmap='gray',
                      vmax=max_value,
                      vmin=max_value*-1,
                      )    

    min_ = -0.5
    max_ = len(networks) + 0.5
    
    ### Draw networks separation lines ###
    network_ticks = [] 
    network_name, indices = np.unique(networks, return_index=True)
    counter = -0.5
    
    colors_ = []
    for net in np.unique(networks):
                    
        items_idx = np.nonzero(networks == net)
        items = items_idx[0].shape[0]
        
        ix = np.nonzero(networks == net)
        if _plot_cfg['ticks_type'] == 'networks':
            tick_ = items_idx[0].mean()
            colors_.append(_plot_cfg['ticks_color'])
        else:
            tick_ = items_idx[0]
            colors_.append(_plot_cfg['ticks_color'][tick_])
            
        counter = counter + items
        
        network_ticks.append(tick_)
        a.axvline(x=counter, ymin=min_, ymax=max_)
        a.axhline(y=counter, xmin=min_, xmax=max_)
    
    
    if _plot_cfg['ticks_type'] == 'networks':
        ticks_labels = np.unique(networks)
    else:
        ticks_labels = roi_names
        
    network_ticks = np.hstack(network_ticks)
    colors_ = np.hstack(colors_)
    
    a.set_yticks(network_ticks)
    a.set_yticklabels(ticks_labels, fontsize=15)
    
    a.set_xticks(network_ticks)
    a.set_xticklabels(ticks_labels, fontsize=15, rotation=80)
    
    
    #colors_[colors_ == facecolor_] = ax_color
    #colors_[colors_ == 'beige'] = 'darkgray'
    
    [t.set_color(colors_[i]) for i,t in enumerate(a.xaxis.get_ticklabels())]
    [t.set_color(colors_[i]) for i,t in enumerate(a.yaxis.get_ticklabels())]
    
    cbar = f.colorbar(ax)

    [t.set_color(ax_color) for i,t in enumerate(cbar.ax.yaxis.get_ticklabels())]
    
    return f


def plot_seaborn(features, node_names=None, node_idx=None, con_thresh=400, **kwargs):
    from mne.viz import circular_layout
    node_angles = circular_layout(node_names.tolist(), 
                                    node_names[node_idx].tolist(), 
                                    start_pos=90, 
                                    group_boundaries=[0, len(node_names) / 2.+1])

    matrix = copy_matrix(array_to_matrix(features.values[0]), diagonal_filler=0.)
    return plot_connectivity_seaborn(matrix, 
                                           node_names=node_names, 
                                           con_thresh=con_thresh, 
                                           node_angles=node_angles, 
                                           node_colors=sns.dark_palette(kwargs['color']))
    




# TODO : Remove tons of plot_connectivity_circle_foo
def plot_connectivity_circle_edited(con, node_names, indices=None, n_lines=None,
                                     node_angles=None, node_width=None,
                                     node_size=None, node_thresh=None, con_thresh=None,
                                     node_colors=None, facecolor='black',
                                     textcolor='white', node_edgecolor='black',
                                     linewidth=1.5, colormap='hot', vmin=None,
                                     vmax=None, colorbar=True, title=None,
                                     colorbar_size=0.2, colorbar_pos=(-0.3, 0.1),
                                     fontsize_title=12, fontsize_names=10,
                                     fontsize_colorbar=8, padding=6.,
                                     fig=None, subplot=111, interactive=False):
    
    """Visualize connectivity as a circular graph, using circles to identify ROIs, their size
    is proportional to the sum of the connections it has in absolute value.

    Note: This code is based on the circle graph example by Nicolas P. Rougier
    http://www.loria.fr/~rougier/coding/recipes.html

    Parameters
    ----------
    con : array
        Connectivity scores. Can be a square matrix, or a 1D array. If a 1D
        array is provided, "indices" has to be used to define the connection
        indices.
    node_names : list of str
        Node names. The order corresponds to the order in con.
    indices : tuple of arrays | None
        Two arrays with indices of connections for which the connections
        strenghts are defined in con. Only needed if con is a 1D array.
    n_lines : int | None
        If not None, only the n_lines strongest connections (strength=abs(con))
        are drawn.
    node_angles : array, shape=(len(node_names,)) | None
        Array with node positions in degrees. If None, the nodes are equally
        spaced on the circle. See mne.viz.circular_layout.
    node_width : float | None
        Width of each node in degrees. If None, "360. / len(node_names)" is
        used.
    node_size : array, shape=(len(node_names,)) | None
        Size of the circles.
    con_thresh : float | None
        Threshold of the connection used to exclude arcs with a smaller absolute value
        to be plotted.
    node_colors : list of tuples | list of str
        List with the color to use for each node. If fewer colors than nodes
        are provided, the colors will be repeated. Any color supported by
        matplotlib can be used, e.g., RGBA tuples, named colors.
    facecolor : str
        Color to use for background. See matplotlib.colors.
    textcolor : str
        Color to use for text. See matplotlib.colors.
    node_edgecolor : str
        Color to use for lines around nodes. See matplotlib.colors.
    linewidth : float
        Line width to use for connections.
    colormap : str
        Colormap to use for coloring the connections.
    vmin : float | None
        Minimum value for colormap. If None, it is determined automatically.
    vmax : float | None
        Maximum value for colormap. If None, it is determined automatically.
    colorbar : bool
        Display a colorbar or not.
    title : str
        The figure title.
    colorbar_size : float
        Size of the colorbar.
    colorbar_pos : 2-tuple
        Position of the colorbar.
    fontsize_title : int
        Font size to use for title.
    fontsize_names : int
        Font size to use for node names.
    fontsize_colorbar : int
        Font size to use for colorbar.
    padding : float
        Space to add around figure to accommodate long labels.
    fig : None | instance of matplotlib.pyplot.Figure
        The figure to use. If None, a new figure with the specified background
        color will be created.
    subplot : int | 3-tuple
        Location of the subplot when creating figures with multiple plots. E.g.
        121 or (1, 2, 1) for 1 row, 2 columns, plot 1. See
        matplotlib.pyplot.subplot.
    interactive : bool
        When enabled, left-click on a node to show only connections to that
        node. Right-click shows all connections.

    Returns
    -------
    fig : instance of matplotlib.pyplot.Figure
        The figure handle.
    axes : instance of matplotlib.axes.PolarAxesSubplot
        The subplot handle.
    """
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches
    from mne.externals.six import string_types
    from functools import partial
    import seaborn as sns
    
    
    n_nodes = len(node_names)

    if node_angles is not None:
        if len(node_angles) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_angles = node_angles * np.pi / 180
    else:
        # uniform layout on unit circle
        node_angles = circular_layout(node_names.tolist(), 
                                      node_names.tolist(), 
                                      start_pos=90,
                                      group_sep=2)
        node_angles = node_angles * np.pi / 180


    if node_width is None:
        node_width = 2 * np.pi / n_nodes
    else:
        node_width = node_width * np.pi / 180


    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
            node_colors = [next(node_colors) for _ in range(n_nodes)]
    else:
        # assign colors using colormap
        node_colors = [plt.cm.winter(i / float(n_nodes))
                       for i in range(n_nodes)]

    
    if node_size is None:
        node_size = 20. + (50*(con.sum(0) / (con.sum(0).max()) * 2.3) )** 1.2

    
    if not isinstance(node_size, np.ndarray):
        node_size = np.full(len(node_names), node_size)
    
    # handle 1D and 2D connectivity information
    if con.ndim == 1:
        if indices is None:
            raise ValueError('indices has to be provided if con.ndim == 1')
    elif con.ndim == 2:
        if con.shape[0] != n_nodes or con.shape[1] != n_nodes:
            raise ValueError('con has to be 1D or a square matrix')
        # we use the lower-triangular part
        indices = np.tril_indices(n_nodes, -1)
        con = con[indices]
    else:
        raise ValueError('con has to be 1D or a square matrix')


    # Draw lines between connected nodes, only draw the strongest connections
    if n_lines is not None and len(con) > n_lines:
        if con_thresh != None:
            con_thresh = np.sort(np.abs(con).ravel())[-n_lines]
    else:
        if con_thresh == None:
            con_thresh = 0.
    
        
    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first

    # This is to plot in gray the lower connections
    draw_thresh = con_thresh / 1.5
    
    con_abs = np.abs(con)
    con_draw_idx = np.where(con_abs >= draw_thresh)[0]
    #con_draw_idx = np.where(con_abs >= con_thresh)[0]

    con = con[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    con = con[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling
    if vmin is None:
        vmin = np.min(con[np.abs(con) >= con_thresh])
    if vmax is None:
        vmax = np.max(con)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.25 * node_width
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    
  
    # get the colormap
    if isinstance(colormap, string_types):
        str_cmap = colormap
        colormap = plt.get_cmap(colormap)
        
    axes = plt.gca()
    
    # Make figure background the same colors as axes
    if fig is None:
        fig = plt.figure(figsize=(12, 12), facecolor=facecolor)
    

    # Use a polar axes
    if not isinstance(subplot, tuple):
        subplot = (subplot,)
    axes = plt.subplot(*subplot, polar=True, facecolor=facecolor)
    
    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additonal space if requested
    #plt.ylim(0, 10 + padding)
    
    
    
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start])
                           / float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end])
                         / float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (con - vmin) / vrange
    con_thresh_scaled = (con_thresh - vmin) / vrange
    
    # Finally, we draw the connections
    nodes = []
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):
        # Start point
        t0, r0 = node_angles[i], 1.2

        # End point
        t1, r1 = node_angles[j], 1.2

        # Some noise in start and end point
        t0 += start_noise[pos]
        t1 += end_noise[pos]

        verts = [(t0, r0), 
                 (t0, 0.5), 
                 (t1, 0.5), 
                 (t1, r1)]
        codes = [m_path.Path.MOVETO, 
                 m_path.Path.CURVE4, 
                 m_path.Path.CURVE4,
                 m_path.Path.LINETO]
        path = m_path.Path(verts, codes)
        
        if np.abs(con[pos]) <= con_thresh:
            #colormap = plt.get_cmap('gray')
            alpha=0.4
            mult = 1.2
        else:
            nodes.append([i,j])
            
            if isinstance(colormap, string_types):
                str_cmap = colormap
                colormap = plt.get_cmap(colormap)

            #colormap = plt.get_cmap(str_cmap)
            alpha=0.8
            mult = np.abs(con[pos])/con.max() * 3

        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=linewidth**mult, alpha=alpha, zorder=0)
        axes.add_patch(patch)

    # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.2
    
    if len(nodes) > 0:
        nodes = np.unique(np.hstack(nodes))
    
    for i, (x,y) in enumerate(zip(node_angles, height)):
        cmap = sns.dark_palette(node_colors[i], n_colors=15, as_cmap=True, reverse=True)
        _ = axes.scatter(x, 
                            y, 
                            s=node_size[i], 
                            c=cmap(node_size[i]/node_size.sum()), 
                            zorder=1, 
                            alpha=0.9, 
                            linewidths=2, 
                            facecolor='.9')
    axes.set_ylim(0, 2)
    
    '''
    bars = axes.bar(node_angles, height, width=node_width, bottom=9,
                    edgecolor=node_edgecolor, lw=2, facecolor='.9',
                    align='center')
    
    for bar, color in zip(bars, node_colors):
        bar.set_facecolor(color)
    '''
    # Draw node labels
    angles_deg = 180 * node_angles / np.pi
    for i, (name, angle_rad, angle_deg, n_size) in enumerate(zip(node_names, node_angles, angles_deg, node_size)):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'
        
        txt_size = fontsize_names + 2
        txt_color = textcolor
        
        if node_thresh is not None:           
            
            
            if n_size > node_thresh:
                txt_color = textcolor
                txt_size = fontsize_names + 3
    
            else:
                txt_color = 'gray'
                txt_size = fontsize_names - 3
        
        else:
            
            if i not in nodes:
                txt_color = 'gray'
                txt_size = fontsize_names - 3                
        
        
        
        
        axes.text(angle_rad, 1.25, name, size=txt_size,
          rotation=angle_deg, rotation_mode='anchor',
          horizontalalignment=ha, verticalalignment='center',
          color=txt_color)

    if title is not None:
        plt.title(title, color=textcolor, fontsize=fontsize_title,
                  axes=axes)

    if colorbar:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array(np.linspace(vmin, vmax))
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                          shrink=colorbar_size,
                          anchor=colorbar_pos)
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize_colorbar)
        plt.setp(cb_yticks, color=textcolor)

    

    return fig, axes



def get_circle_vert(i, j, start_noise, end_noise, pos, node_angles):
    import matplotlib.path as m_path
    t0, r0 = node_angles[i], 1.2

    # End point
    t1, r1 = node_angles[j], 1.2

    # Some noise in start and end point
    t0 += start_noise[pos]
    t1 += end_noise[pos]

    verts = [(t0, r0), 
             (t0, 0.5), 
             (t1, 0.5), 
             (t1, r1)
            ]

    codes = [m_path.Path.MOVETO, 
             m_path.Path.CURVE4, 
             m_path.Path.CURVE4,
             m_path.Path.LINETO
            ]
    
    path = m_path.Path(verts, codes)

    return verts, codes




def get_linear_vert(i, j, start_noise, end_noise, pos, node_angles):
    import matplotlib.path as m_path
    t0, r0 = 1.2, node_angles[i]

    # End point
    t1, r1 = 1.2, node_angles[j], 

    # Some noise in start and end point
    r0 += start_noise[pos]
    r1 += end_noise[pos]

    f = np.random.randn()*0.3
    p0 = 0.7
    
    verts = [(t0, r0), 
            (t0+p0+f, r0), 
            (t1+p0+f, r1), 
            (t1, r1)]

    codes = [m_path.Path.MOVETO, 
                m_path.Path.CURVE4, 
                m_path.Path.CURVE4,
                m_path.Path.LINETO]


    return verts, codes


def get_multi_vert(i, j, start_noise, end_noise, pos, node_angles):
    
    import matplotlib.path as m_path
    # Start point
    t0, r0 = 1.2, node_angles[i]

    # End point
    t1, r1 = 1.2, node_angles[j], 

    # Some noise in start and end point
    r0 += start_noise[pos]
    r1 += end_noise[pos]
    
    verts = [(t0, r0), 
                (t0+.25, r0),
                (t1+.5, r1),
                (t1+1., r1), 
                (t1+1., r1)]

    codes = [m_path.Path.MOVETO, 
             m_path.Path.CURVE4,
             m_path.Path.CURVE3,
             m_path.Path.CURVE4,
             m_path.Path.MOVETO
             ]


    if i == j:
        r0 -= start_noise[pos]
        verts = [(t0, r0), 
                 (t0+1., r0)
                ]

        codes = [
                 m_path.Path.MOVETO, 
                 m_path.Path.LINETO
                ]

    return verts, codes


def plot_connectivity_lines(matrix, 
                            node_names, 
                            kind='circle', 
                            node_position=None, 
                            node_colors=None, 
                            con_thresh=None,
                            linewidth=None,
                            facecolor='black',
                            colormap='magma',
                            font="Manjari",
                            fontsize=14,
                            colorbar=None,
                            title=None,
                            fig=None):
    
    
    import matplotlib.pyplot as plt
    import matplotlib.path as m_path
    import matplotlib.patches as m_patches
    import seaborn as sns
    from sklearn.preprocessing import minmax_scale

    verts_fx = {'multi' : get_multi_vert,
                'linear': get_linear_vert,
                'circle': get_circle_vert}
    
    
    n_nodes = len(node_names)

    if node_position is not None:
        if len(node_position) != n_nodes:
            raise ValueError('node_angles has to be the same length '
                             'as node_names')
        # convert it to radians
        node_position = node_position * np.pi / 180
    else:
        # uniform layout on unit circle
            
        node_position = circular_layout(node_names, 
                                      node_names, 
                                      start_pos=90,
                                      group_sep=0.,
                                      group_boundaries=range(len(node_names)))
        node_position = node_position * np.pi / 180

    if kind != 'circle':
        node_position = np.linspace(0, n_nodes, n_nodes, endpoint=False)

    if node_colors is not None:
        if len(node_colors) < n_nodes:
            node_colors = cycle(node_colors)
            node_colors = [next(node_colors) for _ in range(n_nodes)]
    else:
        # assign colors using colormap
        node_colors = [plt.cm.winter(i / float(n_nodes))
                       for i in range(n_nodes)]

    node_size = minmax_scale(np.abs(matrix).sum(0), feature_range=(0, 30)) ** 2.1 + 150

    k = -1
    if kind == 'multi':
        k = 0

    # handle 1D and 2D connectivity information
    if matrix.shape[0] != n_nodes or matrix.shape[1] != n_nodes:
        raise ValueError('con has to be 1D or a square matrix')
    # we use the lower-triangular part
    indices = np.tril_indices(n_nodes, k)
    matrix = matrix[indices]

    
    # Draw lines between connected nodes, only draw the strongest connections
    if con_thresh == None:
            con_thresh = 0.
    
    textcolor = 'white'
    if facecolor == 'white':
        textcolor = 'black'
    
    
    # get the connections which we are drawing and sort by connection strength
    # this will allow us to draw the strongest connections first

    # This is to plot in gray the lower connections
    draw_thresh = con_thresh / 1.5
    
    con_abs = np.abs(matrix)
    con_draw_idx = np.where(con_abs >= draw_thresh)[0]
    #con_draw_idx = np.where(con_abs >= con_thresh)[0]

    matrix = matrix[con_draw_idx]
    con_abs = con_abs[con_draw_idx]
    indices = [ind[con_draw_idx] for ind in indices]

    # now sort them
    sort_idx = np.argsort(con_abs)
    con_abs = con_abs[sort_idx]
    matrix = matrix[sort_idx]
    indices = [ind[sort_idx] for ind in indices]

    # Get vmin vmax for color scaling

    vmin = np.min(matrix[np.abs(matrix) >= con_thresh])
    vmax = np.max(matrix)
    vrange = vmax - vmin

    # We want to add some "noise" to the start and end position of the
    # edges: We modulate the noise with the number of connections of the
    # node and the connection strength, such that the strongest connections
    # are closer to the node center
    nodes_n_con = np.zeros((n_nodes), dtype=np.int)
    for i, j in zip(indices[0], indices[1]):
        nodes_n_con[i] += 1
        nodes_n_con[j] += 1

    # initalize random number generator so plot is reproducible
    rng = np.random.mtrand.RandomState(seed=0)

    n_con = len(indices[0])
    noise_max = 0.5 * np.pi / n_nodes
    start_noise = rng.uniform(-noise_max, noise_max, n_con)
    end_noise = rng.uniform(-noise_max, noise_max, n_con)

    nodes_n_con_seen = np.zeros_like(nodes_n_con)
    
      # get the colormap
    if facecolor == 'white' and colormap == 'magma':
        colormap = "magma_r"

    if isinstance(colormap, str):
        str_cmap = colormap
        colormap = plt.get_cmap(colormap)

    if isinstance(colormap, sns.palettes._ColorPalette):
        from matplotlib.colors import ListedColormap
        colormap = ListedColormap(colormap.as_hex())
        

    # Make the figure larger for linear plots
    if kind != 'circle':
        figy = n_nodes / 10 + 12
        figx = figy + 3
    else:
        figy = n_nodes / 10 + 5
        figx = figy + 3


    if fig is None:
        fig = plt.figure(figsize=(figx, figy), 
                        facecolor=facecolor)
    
    polar = False
    if kind == 'circle':
        polar = True
    
    # Use a polar axes
    axes = plt.subplot(111, polar=polar, facecolor=facecolor)

    # No ticks, we'll put our own
    plt.xticks([])
    plt.yticks([])

    # Set y axes limit, add additonal space if requested
    #plt.ylim(0, 10 + padding)
    
    for i, (start, end) in enumerate(zip(indices[0], indices[1])):
        nodes_n_con_seen[start] += 1
        nodes_n_con_seen[end] += 1

        start_noise[i] *= ((nodes_n_con[start] - nodes_n_con_seen[start])
                           / float(nodes_n_con[start]))
        end_noise[i] *= ((nodes_n_con[end] - nodes_n_con_seen[end])
                         / float(nodes_n_con[end]))

    # scale connectivity for colormap (vmin<=>0, vmax<=>1)
    con_val_scaled = (matrix - vmin) / vrange
    con_thresh_scaled = (con_thresh - vmin) / vrange
    
    if linewidth is None:
        linewidth = minmax_scale(np.abs(matrix))
    else:
        linewidth *= np.ones_like(con_val_scaled)

    # Finally, we draw the connections
    nodes = []
    for pos, (i, j) in enumerate(zip(indices[0], indices[1])):

        verts, codes = verts_fx[kind](i, j, start_noise, end_noise, pos, node_position)

        path = m_path.Path(verts, codes)
        
        if np.abs(matrix[pos]) <= con_thresh:
            #colormap = plt.get_cmap('gray')
            alpha=0.4
            mult=0
        else:
            nodes.append([i,j])
            #colormap = plt.get_cmap(str_cmap)
            alpha=0.8
            mult=10


        color = colormap(con_val_scaled[pos])

        # Actual line
        patch = m_patches.PathPatch(path, fill=False, edgecolor=color,
                                    linewidth=mult*linewidth[pos], alpha=alpha, zorder=0)
        axes.add_patch(patch)

        # Draw ring with colored nodes
    height = np.ones(n_nodes) * 1.2
    
    nodes = np.unique(np.hstack(nodes))
    
    for i, (x,y) in enumerate(zip(node_position, height)):
        cmap = sns.dark_palette(node_colors[i], n_colors=15, as_cmap=True, reverse=True)

        point = {'x':x, 'y':y }
        if kind != 'circle':
            point['x'] = y
            point['y'] = x

        c = cmap(node_size[i]/node_size.sum())
        c = np.array(c).reshape((-1, len(c)))

        if kind == 'multi':
            _ = axes.scatter(point['x']+1.0, 
                             point['y'], 
                             s=node_size[i], 
                             c=c, 
                             zorder=1, 
                             #alpha=0.9, 
                             linewidths=2, 
                             facecolor='.9')



        
        _ = axes.scatter(point['x'], 
                        point['y'], 
                        s=node_size[i], 
                        c=c, 
                        zorder=1, 
                        #alpha=0.9, 
                        linewidths=2, 
                        facecolor='.9'
                        )
            

    if kind == 'circle':
        axes.set_ylim(0, 1.45)

    else:
        axes.set_xlim(0.8, 2.6)
        #axes.set_ylim(-5, n_nodes+5)
    
    axes.axis('off')

    angles_deg = 180 * node_position / np.pi

    node_ordered = np.argsort(node_size)[::-1]
    
    for i, (name, angle_rad, angle_deg, n_size) in enumerate(zip(node_names, node_position, angles_deg, node_size)):
        if angle_deg >= 270:
            ha = 'left'
        else:
            # Flip the label, so text is always upright
            angle_deg += 180
            ha = 'right'
        
            
        txt_size = fontsize + 2
        txt_color = textcolor
        # Write only big names!
        if i not in nodes:
            txt_color = 'gray'
            txt_size = fontsize - 3

        # Highlight more the higher nodes
        if i in node_ordered[:6]:
            txt_color = 'crimson'
            txt_size = fontsize + 7.5

        if kind == 'circle':
            axes.text(angle_rad, 1.27, name, 
                      size=txt_size,
                      rotation=angle_deg, 
                      rotation_mode='anchor',
                      horizontalalignment=ha, 
                      verticalalignment='center',
                      fontname=font,
                      color=txt_color)

        else:
   
            axes.text(1.2-0.05, 
                        angle_rad, 
                        name, 
                        size=txt_size,
                        #rotation=angle_deg, 
                        #rotation_mode='anchor',
                        horizontalalignment='right', 
                        verticalalignment='center',
                        fontname=font,
                        color=txt_color)
        
        if kind == 'multi':

            axes.text(1.2+1.+0.05, 
                        angle_rad, 
                        name, 
                        size=txt_size,
                        #rotation=angle_deg, 
                        #rotation_mode='anchor',
                        horizontalalignment='left', 
                        verticalalignment='center',
                        fontname=font,
                        color=txt_color)
    
    
    
    if title is not None:
        plt.title(title, 
                  color='gray', 
                  fontsize=fontsize+7,
                  #loc='left',
                  pad=20,
                  axes=axes)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    ticks = np.linspace(vmin, vmax, 4)
    sm.set_array(np.linspace(vmin, vmax))
    
    if colorbar is not None:
        cb = plt.colorbar(sm, ax=axes, use_gridspec=False,
                            orientation='vertical', pad=0.1,
                            shrink=0.25,
                            #ticks=ticks,
                            #anchor=colorbar_pos
                            # 
                            )
        cb_yticks = plt.getp(cb.ax.axes, 'yticklabels')
        cb.ax.tick_params(labelsize=fontsize-2)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_family(font)
        plt.setp(cb_yticks, color=textcolor)

    return fig
    



def plot_connectome(matrix, 
                    coords, 
                    colors, 
                    size, 
                    threshold, 
                    fname,                    
                    cmap=plt.cm.hot, 
                    title='', 
                    max_=None, 
                    min_=None, 
                    display_='ortho'):
    
    """
    Wrapper of the plot_connectome function in nilearn with some fixed
    values
    """
    
    from nilearn import plotting
    
    plotting.plot_connectome(adjacency_matrix=matrix, 
                             node_coords=coords, 
                             node_color=colors.tolist(), 
                             node_size=1.5*size, 
                             edge_cmap=cmap, 
                             edge_vmin=min_, 
                             edge_vmax=max_, 
                             edge_threshold=threshold, 
                             output_file=fname, 
                             display_mode=display_, 
                             figure=plt.figure(figsize=(16*1.2,9*1.2)),# facecolor='k', edgecolor='k'), 
                             #axes, 
                             title=title, 
                             #annotate, 
                             black_bg=True, 
                             #alpha, 
                             edge_kwargs={
                                          'alpha':0.8,
                                          'linewidth':9,
                                          }, 
                             node_kwargs={
                                          'edgecolors':'k',
                                          }, 
                             #colorbar=True
                             )
    



def plot_connectomics(matrix, 
                      node_size, 
                      save_path, 
                      prename,
                      save=False,
                      **kwargs
                      ):
    
    
    
    _plot_cfg = {
                 'threshold':1.4,
                 'fontsize_title':19,
                 'fontsize_colorbar':13,
                 'fontsize_names':13,
                 'colorbar_size':0.3,
                 'colormap':'hot',
                 'vmin':-3,
                 'vmax':3,
                 'figure':plt.figure(figsize=(16,16)),
                 'facecolor':'black',
                 'dpi':150,
                 'name':'weights',
                 'title':'Connectome',
                 'filetype':'png',
                 'zscore': True      
                }
    
    
    
    _plot_cfg.update(kwargs)
     
    directory_ = save_path[save_path.rfind('/')+1:]
    
    #names_lr, colors_lr, index_, coords = get_plot_stuff(directory_)
    
    names_lr = kwargs['node_names']
    colors_lr = kwargs['node_colors']
    index_ = kwargs['node_order']
    coords = kwargs['node_coords']
    networks = kwargs['networks']
    
    matrix = matrix[index_][:,index_]
    names_lr = names_lr[index_]
    node_colors = colors_lr[index_]
    node_size = node_size[index_]
    
    f, _ = plot_connectivity_circle_edited(matrix, 
                                            names_lr, 
                                            node_colors=node_colors,
                                            node_size=node_size,
                                            con_thresh=_plot_cfg['threshold'],
                                            title=_plot_cfg['title'],
                                            node_angles=circular_layout(names_lr, 
                                                                        list(names_lr),
                                                                        ),
                                            fontsize_title=_plot_cfg['fontsize_title'],
                                            fontsize_names=_plot_cfg['fontsize_names'],
                                            fontsize_colorbar=_plot_cfg['fontsize_colorbar'],
                                            colorbar_size=_plot_cfg['colorbar_size'],
                                            colormap=_plot_cfg['colormap'],
                                            vmin=_plot_cfg['vmin'],
                                            vmax=_plot_cfg['vmax'],
                                            fig=_plot_cfg['figure'],
                                            )
            
    if save == True:
        fname = "%s_features_%s.%s" % (prename, _plot_cfg['name'], _plot_cfg['filetype'])
        
        f.savefig(os.path.join(save_path, fname),
                          facecolor=_plot_cfg['facecolor'],
                          dpi=_plot_cfg['dpi'])
    
    
    for d_ in ['x', 'y', 'z']:
        
        fname = None
        if save == True:
            fname = "%s_connectome_feature_%s_%s.%s" %(prename, 
                                                       _plot_cfg['name'], 
                                                       d_,
                                                       _plot_cfg['filetype'])
            fname = os.path.join(save_path, fname)
            
        plot_connectome(matrix, 
                        coords, 
                        colors_lr, 
                        node_size,
                        _plot_cfg['threshold'],
                        fname,
                        cmap=_plot_cfg['colormap'],
                        title=None,
                        display_=d_,
                        max_=_plot_cfg['vmax'],
                        min_=_plot_cfg['vmin']
                        )
        

    f = plot_matrix(matrix, _, networks, threshold=_plot_cfg['threshold'],
                                            zscore=_plot_cfg['zscore'])
    
    if save == True:
        fname = "%s_matrix_%s.%s" %(prename, _plot_cfg['name'], _plot_cfg['filetype'])
        f.savefig(os.path.join(save_path, fname),
                          facecolor=_plot_cfg['facecolor'],
                          dpi=_plot_cfg['dpi'])





def plot_regression_errors(errors, 
                           permutation_error, 
                           save_path, 
                           prename='distribution', 
                           errors_label=['MSE','COR']):
    
    fig_ = plt.figure()
    bpp = plt.boxplot(permutation_error, showfliers=False, showmeans=True, patch_artist=True)
    bpv = plt.boxplot(errors, showfliers=False, showmeans=True, patch_artist=True)
    fname = "%s_perm_1000_boxplot.png" %(prename)
   
    
    for box_, boxp_ in zip(bpv['boxes'], bpp['boxes']):
        box_.set_facecolor('lightgreen')
        boxp_.set_facecolor('lightslategrey')
      
      
    plt.xticks(np.array([1,2]), errors_label)
    
    plt.savefig(os.path.join(save_path, fname))
    plt.close()
    
    return fig_


def plot_within_between_weights(connections,
                                condition,
                                savepath,
                                atlas='findlab', 
                                background='white'):
    
    import matplotlib.pyplot as pl
    names_lr, colors_lr, index_, coords, networks = get_atlas_info(atlas, background=background)
    _, idxnet = np.unique(networks, return_index=True)
    _, idx = np.unique(colors_lr, return_index=True)
    
    color_net = dict(zip(networks[np.sort(idxnet)], colors_lr[np.sort(idx)]))

    fig = pl.figure(figsize=(13.2,10), dpi=200)
    
    for k_, v_ in connections.iteritems():
        lines_ = [pl.plot(v_, 'o-', c=color_net[k_], 
                          markersize=20, linewidth=5, alpha=0.6, 
                          label=k_)]
         
    
    pl.legend()
    pl.ylabel("Average connection weight")
    pl.xticks([0,1,1.4], ['Between-Network', 'Within-Network',''])
    pl.title(condition+' within- and between-networks average weights')
    pl.savefig(os.path.join(savepath, condition+'_decoding_within_between.png'),
               dpi=200)

    
    return fig



def plot_features_distribution(feature_set, 
                               feature_set_permutation, 
                               save_path, 
                               prename='features', 
                               n_features=90, 
                               n_bins=20):
    
    plt.figure()
    h_values_p, _ = np.histogram(feature_set_permutation.flatten(), 
                                 bins=np.arange(0, n_features+1))
    
    plt.hist(zscore(h_values_p), bins=n_bins)
    
    fname = "%s_features_set_permutation_distribution.png" % (prename)
    plt.savefig(os.path.join(save_path, 
                            fname))
    
    plt.figure()
    h_values_, _ = np.histogram(feature_set.flatten(), 
                                bins=np.arange(0, n_features+1))
    plt.plot(zscore(h_values_))
        
    
    fname = "%s_features_set_cross_validation.png" % (prename)
    plt.savefig(os.path.join(save_path, 
                            fname))

    plt.close('all')
    
    
    
    
    
def plot_cross_correlation(xcorr, t_start, t_end, labels):

    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    dim = len(labels)
    
    fig = plt.figure()
    ax = plt.axes(xlim=(-0.5, dim-0.5), ylim=(dim-0.5, -0.5))
    
    #im = ax.imshow(xcorr.at(t_start), interpolation='nearest', vmin=-1, vmax=1)
    im = ax.imshow(np.eye(dim), interpolation='nearest', vmin=-4, vmax=4)
    title = ax.set_title('')
    xt = ax.set_xticks(np.arange(dim))
    xl = ax.set_xticklabels(labels, rotation='vertical')
    yt = ax.set_yticks(np.arange(dim))
    yl = ax.set_yticklabels(labels)
    fig.colorbar(im)

    l_time = np.arange(-50, 50, 1)
    mask = (l_time >= t_start) * (l_time<=t_end)
    
    def init():
        im.set_array(np.eye(dim))
        title.set_text('Cross-correlation at time lag of '+str(t_start)+' TR.')
        plt.draw()
        return im, title
        
    def animate(i):
        global l_time
        j = np.int(np.rint(i/20))
        #im.set_array(xcorr.at(l_time[j]))
        im.set_array(xcorr[mask][j])
        title.set_text('Cross-correlation at time lag of '+str(l_time[mask][j])+' TR.')
        plt.draw()
        return im, title

    ani = animation.FuncAnimation(fig, animate, 
                                  init_func=init, 
                                  frames=20*(t_end-t_start), 
                                  interval=10,
                                  repeat=False, 
                                  blit=True)
    plt.show()
    #ani.save('/home/robbis/xcorrelation_.mp4')
    
      
def plot_dendrogram(dendrogram, dissimilarity_matrix):
    
    return
