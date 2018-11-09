"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

from ..util.core import *
import numpy as np
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
import string
Path = path.Path
PathPatch = patches.PathPatch


def custom_plt_format():
    """
    Loads the custom plotting format
    """

    # Loads the custom style
    plt.style.use('./custom.mplstyle')


def labelfigs(axes, number, style='wb', loc='br',
              string_add='', size=14, text_pos='center'):
    """
    Adds labels to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    number : int
        letter number
    style : str, optional
        sets the color of the letters
    loc : str, optional
        sets the location of the label
    string_add : str, optional
        custom string as the label
    size : int, optional
        sets the font size for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formatting_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formatting_key[style]

    # finds the position for the label
    x_min, x_max = axes.get_xlim()
    y_min, y_max = axes.get_ylim()
    x_value = .08 * (x_max - x_min) + x_min

    # Sets the location of the label on the figure
    if loc == 'br':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'tr':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = .08 * (x_max - x_min) + x_min
    elif loc == 'bl':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tl':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_max - .08 * (x_max - x_min)
    elif loc == 'tm':
        y_value = y_max - .9 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    elif loc == 'bm':
        y_value = y_max - .1 * (y_max - y_min)
        x_value = x_min + (x_max - x_min) / 2
    else:
        raise ValueError(
            'Unknown string format imported please look at code for acceptable positions')

    # adds a custom string
    if string_add == '':

        # Turns to image number into a label
        if number < 26:
            axes.text(x_value, y_value, string.ascii_lowercase[number],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])

        # allows for double letter index
        else:
            axes.text(x_value, y_value, string.ascii_lowercase[0] + string.ascii_lowercase[number - 26],
                      size=14, weight='bold', ha=text_pos,
                      va='center', color=formatting['color'],
                      path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                           foreground="k")])
    else:
        # writes the text to the figure
        axes.text(x_value, y_value, string_add,
                  size=14, weight='bold', ha=text_pos,
                  va='center', color=formatting['color'],
                  path_effects=[patheffects.withStroke(linewidth=formatting['linewidth'],
                                                       foreground="k")])


def layout_fig(graph, mod=None):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    Parameters
    ----------
    graphs : int
        number of axes to make
    mod : int (optional)
        sets the number of figures per row

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """

    if mod is None:
        # Selects the number of columns to have in the graph
        if graph < 3:
            mod = 2
        elif graph < 5:
            mod = 3
        elif graph < 10:
            mod = 4
        elif graph < 17:
            mod = 5
        elif graph < 26:
            mod = 6
        elif graph < 37:
            mod = 7

    # builds the figure based on the number of graphs and selected number of columns
    fig, axes = plt.subplots(graph // mod + (graph % mod > 0), mod,
                             figsize=(3 * mod, 3 * (graph // mod + (graph % mod > 0))))

    # deletes extra unneeded axes
    axes = axes.reshape(-1)
    for i in range(axes.shape[0]):
        if i + 1 > graph:
            fig.delaxes(axes[i])

    return (fig, axes)


def scalebar(axes, image_size, scale_size, units='nm', loc='br'):
    """
    Adds scalebar to figures

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    image_size : int
        size of the image in nm
    scale_size : str, optional
        size of the scalebar in units of nm
    units : str, optional
        sets the units for the label
    loc : str, optional
        sets the location of the label
    """

    # gets the size of the image
    x_lim, y_lim = axes.get_xlim(), axes.get_ylim()
    x_size, y_size = np.abs(
        np.floor(x_lim[1] - x_lim[0])), np.abs(np.floor(y_lim[1] - y_lim[0]))
    # computes the fraction of the image for the scalebar
    fract = scale_size / image_size

    x_point = np.linspace(x_lim[0], x_lim[1],
                          np.floor(image_size))
    y_point = np.linspace(y_lim[0], y_lim[1],
                          np.floor(image_size))

    # sets the location of the scalebar
    if loc == 'br':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.1 * image_size // 1)]
        y_end = y_point[np.int((.1 + .025) * image_size // 1)]
        y_label_height = y_point[np.int((.1 + .075) * image_size // 1)]
    elif loc == 'tr':
        x_start = x_point[np.int(.9 * image_size // 1)]
        x_end = x_point[np.int((.9 - fract) * image_size // 1)]
        y_start = y_point[np.int(.9 * image_size // 1)]
        y_end = y_point[np.int((.9 - .025) * image_size // 1)]
        y_label_height = y_point[np.int((.9 - .075) * image_size // 1)]

    # makes the path for the scalebar
    path_maker(axes, [x_start, x_end, y_start, y_end], 'w', 'k', '-', 1)

    # adds the text label for the scalebar
    axes.text((x_start + x_end) / 2,
              y_label_height,
              '{0} {1}'.format(scale_size, units),
              size=14, weight='bold', ha='center',
              va='center', color='w',
              path_effects=[patheffects.withStroke(linewidth=1.5,
                                                   foreground="k")])


def set_value_scale(data, clim):
    """
    removes values outside of range

    Parameters
    ----------
    data : array
        data to remove values
    clim : array
        range of values to save

    Return
    ----------
    data : array
        modified data

    """
    data[np.where(data < clim[0])] = clim[0]
    data[np.where(data > clim[1])] = clim[1]
    return data
