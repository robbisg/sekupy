import seaborn as sns
import matplotlib.pyplot as pl


def get_wes_palette(film="rushmore", n_colors=None):

    palette = {
        "BottleRocket_all": ['#A42820', '#5F5647', '#9B110E', '#3F5151',
                             '#4E2A1E', '#550307', '#0C1707', '#FAD510',
                             '#CB2314', '#273046', '#354823', '#1E1E1E'],

        "BottleRocket1": ['#A42820', '#5F5647', '#9B110E', '#3F5151',
                          '#4E2A1E', '#550307', '#0C1707'],

        "BottleRocket2": ['#FAD510', '#CB2314', '#273046',
                          '#354823', '#1E1E1E'],

        "Rushmore_all": ['#E1BD6D', '#EABE94', '#0B775E', '#35274A',
                         '#F2300F', '#E1BD6D', '#EABE94', '#0B775E',
                         '#35274A', '#F2300F'],

        "Rushmore": ['#E1BD6D', '#EABE94', '#0B775E', '#35274A', '#F2300F'],

        "Rushmore1": ['#E1BD6D', '#EABE94', '#0B775E', '#35274A', '#F2300F'],

        "Royal_all": ['#899DA4', '#C93312', '#FAEFD1', '#DC863B', '#9A8822',
                      '#F5CDB4', '#F8AFA8', '#FDDDA0', '#74A089'],

        "Royal1": ['#899DA4', '#C93312', '#FAEFD1', '#DC863B'],

        "Royal2": ['#9A8822', '#F5CDB4', '#F8AFA8', '#FDDDA0', '#74A089'],

        "Zissou1": ['#3B9AB2', '#78B7C5', '#EBCC2A', '#E1AF00', '#F21A00'],

        "Darjeeling_all": ['#FF0000', '#00A08A', '#F2AD00', '#F98400',
                           '#5BBCD6', '#ECCBAE', '#046C9A', '#D69C4E',
                           '#ABDDDE', '#000000'],

        "Darjeeling1": ['#FF0000', '#00A08A', '#F2AD00', '#F98400', '#5BBCD6'],

        "Darjeeling2": ['#ECCBAE', '#046C9A', '#D69C4E', '#ABDDDE', '#000000'],

        "Chevalier1": ['#446455', '#FDD262', '#D3DDDC', '#C7B19C'],

        "FantasticFox1": ['#DD8D29', '#E2D200', '#46ACC8', '#E58601', 
                          '#B40F20'],

        "Moonrise_all": ['#F3DF6C', '#CEAB07', '#D5D5D3', '#24281A', '#798E87',
                         '#C27D38', '#CCC591', '#29211F', '#85D4E3', '#F4B5BD',
                         '#9C964A', '#CDC08C', '#FAD77B'],

        "Moonrise1": ['#F3DF6C', '#CEAB07', '#D5D5D3', '#24281A'],

        "Moonrise2": ['#798E87', '#C27D38', '#CCC591', '#29211F'],

        "Moonrise3": ['#85D4E3', '#F4B5BD', '#9C964A', '#CDC08C', '#FAD77B'],

        "Cavalcanti1": ['#D8B70A', '#02401B', '#A2A475', '#81A88D', '#972D15'],

        "GrandBudapest_all": ['#F1BB7B', '#FD6467', '#5B1A18', '#D67236',
                              '#E6A0C4', '#C6CDF7', '#D8A499', '#7294D4'],

        "GrandBudapest1": ['#F1BB7B', '#FD6467', '#5B1A18', '#D67236', ],

        "GrandBudapest2": ['#E6A0C4', '#C6CDF7', '#D8A499', '#7294D4', ],

        "IsleofDogs_all": ['#9986A5', '#79402E', '#CCBA72', '#0F0D0E',
                           '#D9D0D3', '#8D8680', ],

        "IsleofDogs1": ['#9986A5', '#79402E', '#CCBA72', '#0F0D0E',
                        '#D9D0D3', '#8D8680', '#EAD3BF', '#AA9486',
                        '#B6854D', '#39312F', '#1C1718'],

        "IsleofDogs2": ['#EAD3BF', '#AA9486', '#B6854D',
                        '#39312F', '#1C1718']
    }

    return sns.color_palette(palette[film], n_colors=n_colors)


def plot_colortable(colors, title, sort_colors=True, emptycols=0):
    import matplotlib.colors as mcolors
    
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    colors = {v:v for v in colors}


    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = pl.subplots(figsize=(750 / dpi, 140 / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=18,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=18)

    return fig