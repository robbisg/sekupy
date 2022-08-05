import seaborn as sns
import matplotlib.pyplot as pl


def get_painter_palette(artist, n_colors=None):
    palette = dict(
        austria=list("#a40000", "#16317d", "#007e2f",
                     "#ffcd12", "#b86092", "#721b3e",
                     "#00b7a7"),

        cassatt=list("#b1615c", "#d88782", "#e3aba7",
                     "#edd7d9", "#c9c9dd", "#9d9dc7",
                     "#8282aa", "#5a5a83"),

        cross=list("#c969a1", "#ce4441", "#ee8577",
                   "#eb7926", "#ffbb44", "#859b6c",
                   "#62929a", "#004f63", "#122451"),

        degas=list("#591d06", "#96410e", "#e5a335", "#556219",
                   "#418979", "#2b614e", "#053c29"),

        egypt=list("#dd5129", "#0f7ba2", "#43b284", "#fab255"),

        gauguin=list("#b04948", "#811e18", "#9e4013", "#c88a2c",
                     "#4c6216", "#1a472a"),

        greek=list("#3c0d03", "#8d1c06", "#e67424",
                   "#ed9b49", "#f5c34d"),

        hokusai=list("#6d2f20", "#b75347", "#df7e66", "#e09351",
                     "#edc775", "#94b594", "#224b5e"),

        ingres=list("#041d2c", "#06314e", "#18527e", "#2e77ab",
                    "#d1b252", "#a97f2f", "#7e5522", "#472c0b"),

        isfahan1=list("#4e3910", "#845d29", "#d8c29d", "#4fb6ca",
                      "#178f92", "#175f5d", "#1d1f54"),

        isfahan2=list("#d7aca1", "#ddc000", "#79ad41",
                      "#34b6c6", "#4063a3"),

        juarez=list("#a82203", "#208cc0", "#f1af3a",
                    "#cf5e4e", "#637b31", "#003967"),

        klimt=list("#df9ed4", "#c93f55", "#eacc62",
                   "#469d76", "#3c4b99", "#924099"),

        manet=list("#3b2319", "#80521c", "#d29c44", "#ebc174",
                   "#ede2cc", "#7ec5f4", "#4585b7", "#225e92",
                   "#183571", "#43429b", "#5e65be"),

        monet=list("#4e6d58", "#749e89", "#abccbe", "#e3cacf",
                   "#c399a2", "#9f6e71", "#41507b", "#7d87b2",
                   "#c2cae3"),

        moreau=list("#421600", "#792504", "#bc7524", "#8dadca",
                    "#527baa", "#104839", "#082844"),

        morgenstern=list("#7c668c", "#b08ba5", "#dfbbc8", "#ffc680",
                         "#ffb178", "#db8872", "#a56457"),

        nattier=list("#52271c", "#944839", "#c08e39", "#7f793c",
                     "#565c33", "#184948", "#022a2a"),

        new_kingdom=list("#e1846c", "#9eb4e0", "#e6bb9e",
                         "#9c6849", "#735852"),

        pillement=list("#a9845b", "#697852", "#738e8e", "#44636f",
                       "#2b4655", "#0f252f"),

        pissaro=list("#134130", "#4c825d", "#8cae9e", "#8dc7dc",
                     "#508ca7", "#1a5270", "#0e2a4d"),

        redon=list("#5b859e", "#1e395f", "#75884b", "#1e5a46",
                   "#df8d71", "#af4f2f", "#d48f90", "#732f30",
                   "#ab84a5", "#59385c", "#d8b847", "#b38711"),

        renoir=list("#17154f", "#2f357c", "#6c5d9e", "#9d9cd5",
                    "#b0799a", "#f6b3b0", "#e48171", "#bf3729",
                    "#e69b00", "#f5bb50", "#ada43b", "#355828"),

        robert=list("#11341a", "#375624", "#6ca4a0",
                    "#487a7c", "#18505f", "#062e3d"),

        stevens=list("#042e4e", "#307d7f", "#598c4c",
                     "#ba5c3f", "#a13213", "#470c00"),

        tara=list("#eab1c6", "#d35e17", "#e18a1f", "#e9b109", "#829d44"),

        thomas=list("#b24422", "#c44d76", "#4457a5", "#13315f", "#b1a1cc",
                    "#59386c", "#447861", "#7caf5c"),

        tiepolo=list("#802417", "#c06636", "#ce9344", "#e8b960",
                     "#646e3b", "#2b5851", "#508ea2", "#17486f"),

        troy=list("#421401", "#6c1d0e", "#8b3a2b", "#c27668",
                  "#7ba0b4", "#44728c", "#235070", "#0a2d46"),

        van_gogh1=list("#2c2d54", "#434475", "#6b6ca3", "#969bc7",
                       "#87bcbd", "#89ab7c", "#6f9954"),

        van_gogh2=list("#bd3106", "#d9700e", "#e9a00e", "#eebe04",
                       "#5b7314", "#c3d6ce", "#89a6bb", "#454b87"),

        veronese=list("#67322e", "#99610a", "#c38f16", "#6e948c",
                      "#2c6b67", "#175449", "#122c43"),

        wissing=list("#4b1d0d", "#7c291e", "#ba7233", "#3a4421", "#2d5380")
    )

    return sns.color_palette(palette[artist], n_colors=n_colors)


def get_wes_palette(film="rushmore", n_colors=None):

    palette = {
        "surora": ["#BF616A", "#D08770", "#EBCB8B", "#A3BE8C","#B48EAD"],
        "bottle_rocket_all": ['#A42820', '#5F5647', '#9B110E', '#3F5151',
                             '#4E2A1E', '#550307', '#0C1707', '#FAD510',
                             '#CB2314', '#273046', '#354823', '#1E1E1E'],

        "bottle_rocket1": ['#A42820', '#5F5647', '#9B110E', '#3F5151',
                          '#4E2A1E', '#550307', '#0C1707'],

        "bottle_rocket2": ['#FAD510', '#CB2314', '#273046',
                          '#354823', '#1E1E1E'],

        "rushmore": ['#E1BD6D', '#EABE94', '#0B775E', '#35274A',
                         '#F2300F'],

        "royal_all": ['#899DA4', '#C93312', '#FAEFD1', '#DC863B', '#9A8822',
                      '#F5CDB4', '#F8AFA8', '#FDDDA0', '#74A089'],

        "royal1": ['#899DA4', '#C93312', '#FAEFD1', '#DC863B'],

        "royal2": ['#9A8822', '#F5CDB4', '#F8AFA8', '#FDDDA0', '#74A089'],

        "zissou1": ['#3B9AB2', '#78B7C5', '#EBCC2A', '#E1AF00', '#F21A00'],

        "darjeeling_all": ['#FF0000', '#00A08A', '#F2AD00', '#F98400',
                           '#5BBCD6', '#ECCBAE', '#046C9A', '#D69C4E',
                           '#ABDDDE', '#000000'],

        "darjeeling1": ['#FF0000', '#00A08A', '#F2AD00', '#F98400', '#5BBCD6'],

        "darjeeling2": ['#ECCBAE', '#046C9A', '#D69C4E', '#ABDDDE', '#000000'],

        "chevalier1": ['#446455', '#FDD262', '#D3DDDC', '#C7B19C'],

        "fantastic_fox1": ['#DD8D29', '#E2D200', '#46ACC8', '#E58601', 
                          '#B40F20'],

        "moonrise_all": ['#F3DF6C', '#CEAB07', '#D5D5D3', '#24281A', '#798E87',
                         '#C27D38', '#CCC591', '#29211F', '#85D4E3', '#F4B5BD',
                         '#9C964A', '#CDC08C', '#FAD77B'],

        "moonrise1": ['#F3DF6C', '#CEAB07', '#D5D5D3', '#24281A'],

        "moonrise2": ['#798E87', '#C27D38', '#CCC591', '#29211F'],

        "moonrise3": ['#85D4E3', '#F4B5BD', '#9C964A', '#CDC08C', '#FAD77B'],

        "cavalcanti1": ['#D8B70A', '#02401B', '#A2A475', '#81A88D', '#972D15'],

        "grand_budapest_all": ['#F1BB7B', '#FD6467', '#5B1A18', '#D67236',
                              '#E6A0C4', '#C6CDF7', '#D8A499', '#7294D4'],

        "grand_budapest1": ['#F1BB7B', '#FD6467', '#5B1A18', '#D67236', ],

        "grand_budapest2": ['#E6A0C4', '#C6CDF7', '#D8A499', '#7294D4', ],

        "isle_of_dogs_all": ['#9986A5', '#79402E', '#CCBA72', '#0F0D0E',
                           '#D9D0D3', '#8D8680', ],

        "isle_of_dogs1": ['#9986A5', '#79402E', '#CCBA72', '#0F0D0E',
                        '#D9D0D3', '#8D8680', '#EAD3BF', '#AA9486',
                        '#B6854D', '#39312F', '#1C1718'],

        "isle_of_dogs2": ['#EAD3BF', '#AA9486', '#B6854D',
                        '#39312F', '#1C1718'],

        "french_dispatch1": ['#5f8065', '#881f24', '#944c34', '#bb9d79', '#275066'],
        "french_dispatch2": ['#73a87c', '#c1bc78', '#205d89', '#cf784b'],
        "french_dispatch3": ['#eba2b6', '#e7ccaf', '#292176', '#e0bd59'],
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


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb