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
        ax.tick_params(top='off', 
                       bottom='off', 
                       left='off', 
                       right='off', 
                       labelleft='off', 
                       labelbottom='on')

        for position in ['top', 'left', 'right', 'bottom']:
            ax.spines[position].set_visible(False)


        
    pl.suptitle(title, 
                fontsize=text_size + 10, 
                fontname=font)
    pl.subplots_adjust(top=0.9)

    return fig