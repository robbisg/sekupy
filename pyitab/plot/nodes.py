import matplotlib.colors as plc
import matplotlib.pyplot as pl
import numpy as np
from sklearn.preprocessing import minmax_scale

def get_brightness(c):
    rr, gg, bb = plc.to_rgb(c)
    br = 0.2126*rr + 0.7152*gg + 0.0722*bb
    return br

# TODO : Document
def barplot_nodes(array_list, 
                  names, 
                  colors, 
                  subtitles=None,
                  title=None,
                  selected_nodes=10, 
                  n_rows=1, 
                  n_cols=1, 
                  text_size=25,
                  xmin=0.,
                  font="Manjari"):

    fig = pl.figure(figsize=(18, 15))
    # selected_nodes = 15
    y_pos = range(selected_nodes)

    for i, magnitude in enumerate(array_list):
        arg_ = np.argsort(magnitude)

        norm_size  = minmax_scale(magnitude)
        norm_size  = norm_size[arg_][::-1][:selected_nodes]

        sort_size  = magnitude[arg_][::-1][:selected_nodes]
        sort_color = colors[arg_][::-1][:selected_nodes]
        sort_names = names[arg_][::-1][:selected_nodes]
        
        #norm_size = minmax_scale(magnitude) + 1
             
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        for j in y_pos:
            ax.barh(selected_nodes-j-1, 
                    norm_size[j], 
                    align='center', 
                    color=sort_color[j], 
                    #edgecolor='k',
                    #lw=2.5
                )
            br = get_brightness(sort_color[j])
            cc = 'k' if br > 0.5 else 'white'
            label = ax.text(norm_size[j]-0.0075, 
                            selected_nodes-j-1, 
                            sort_names[j],
                            verticalalignment='center',
                            horizontalalignment='right',
                            #weight='bold',
                            color=cc,
                            clip_on=True, 
                            fontsize=text_size,
                            fontname=font)

            value = ax.text(norm_size[j]+0.0075, 
                            selected_nodes-j-1, 
                            str(sort_size[j])[:6], 
                            verticalalignment='center', 
                            #weight='bold',
                            fontname=font,
                            fontsize=text_size
                            )
                            
        ax.set_yticks(np.array(y_pos)[::-1])
        ax.set_yticklabels([])
        ax.set_xlim([xmin, 1.])
        ax.set_xticklabels([],[])

        ax.tick_params(axis='both', 
                        which='major',
                        labelsize=text_size-5)

        #ax.set_xticklabels(ax.xaxis.get_major_ticks(), fontsize=20)
        if subtitles is not None:
            ax.set_title(subtitles[i], 
                        fontsize=text_size + 3,
                        fontname=font,
                        )
        """
        ax.set_xlabel("Average selection probability", 
                      fontsize=text_size + 3,
                      fontname=font,
                      )
        """
        ax.tick_params(top=False, 
                       bottom=False, 
                       left=False, 
                       right=False, 
                       labelleft=False, 
                       labelbottom=True)

        for position in ['top', 'left', 'right', 'bottom']:
            ax.spines[position].set_visible(False)


        
    pl.suptitle(title, 
                fontsize=text_size + 10, 
                fontname=font)
    pl.subplots_adjust(top=0.9)

    return fig




def scatter_nodes(array_list, 
                  names, 
                  colors, 
                  subtitles=None,
                  title=None,
                  selected_nodes=15, 
                  n_rows=1, 
                  n_cols=1, 
                  text_size=25,
                  xmin=0.,
                  font="Manjari"):

    import matplotlib.path as m_path
    import matplotlib.patches as m_patches

    indices = np.array([np.argsort(s) for s in array_list])

    fig = pl.figure()
    ax = fig.add_subplot(111)


    for i in range(len(indices[0])):
        x, y = np.nonzero(indices == i)

        ax.scatter(x, y, s=np.array(array_list)[x,y], c='gray')

        highest = y > 99 - selected_nodes
        ax.scatter(x[highest], y[highest], s=85)

        if np.any(highest):

            for i in range(len(y)-1):

                t0, r0 = x[i], y[i]
                t1, r1 = x[i], y[i+1]

                if r0 > 99 - selected_nodes and r1 > 99 - selected_nodes:
                    alpha=1
                    linewidth=2.5
                    color='red'
                else:
                    alpha=0.4
                    linewidth=2
                    color='red'

                alpha=1
                linewidth=2.5
                color='red'
               
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


                if r0 == r1:
                    #r0 -= start_noise[pos]
                    verts = [(t0, r0), 
                            (t0+1., r0)
                            ]

                    codes = [
                            m_path.Path.MOVETO, 
                            m_path.Path.LINETO
                            ]

                path = m_path.Path(verts, codes)


                patch = m_patches.PathPatch(path, fill=False, color=color, linewidth=linewidth,
                                                alpha=alpha, zorder=0)
                ax.add_patch(patch)