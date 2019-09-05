"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

from scipy import special
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
import numpy as np
from matplotlib import (pyplot as plt, path, patches)

Path = path.Path
PathPatch = patches.PathPatch
erf = special.erf


def path_maker(axes, locations, facecolor, edgecolor, linestyle, lineweight):
    """
    Adds path to figure

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    locations : numpy array
        location to position the path
    facecolor : str, optional
        facecolor of the path
    edgecolor : str, optional
        edgecolor of the path
    linestyle : str, optional
        sets the style of the line, using conventional matplotlib styles
    lineweight : float, optional
        thickness of the line
    """
    vertices = []
    codes = []
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    # extracts the vertices used to construct the path
    vertices = [(locations[0], locations[2]),
                (locations[1], locations[2]),
                (locations[1], locations[3]),
                (locations[0], locations[3]),
                (0, 0)]
    vertices = np.array(vertices, float)
    #  makes a path from the vertices
    path = Path(vertices, codes)
    pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,
                          ls=linestyle, lw=lineweight)
    # adds path to axes
    axes.add_patch(pathpatch)


def rotate_and_crop(image_, angle=60.46, frac_rm=0.17765042979942694):
    """
    Function which rotates and crops the images

    Parameters
    ----------
    image_ : array
        image array to plot
    angle  : float, optional
        angle to rotate the image by
    frac_rm  : float, optional
        sets the fraction of the image to remove

    Returns
    ----------
    crop_image : array
        image which is rotated and cropped
    scale_factor : float
        scaling factor for the image following rotation
    """
    # makes a copy of the image
    image = np.copy(image_)
    # replaces all points with the minimum value
    image[~np.isfinite(image)] = np.nanmin(image)
    # rotates the image
    rot_topo = ndimage.interpolation.rotate(
        image, 90-angle, cval=np.nanmin(image))
    # crops the image
    pix_rem = int(rot_topo.shape[0]*frac_rm)
    crop_image = rot_topo[pix_rem:rot_topo.shape[0] -
                          pix_rem, pix_rem:rot_topo.shape[0]-pix_rem]
    # returns the scale factor for the new image size
    scale_factor = (np.cos(np.deg2rad(angle)) +
                    np.cos(np.deg2rad(90-angle)))*(1-frac_rm)

    return crop_image, scale_factor


def roll_and_append(data, fraction=4):
    """
    Function which rotates and crops the images

    Parameters
    ----------
    data : array
        input data to process
    fraction  : float, optional
        fraction to roll and append

    Returns
    ----------
    data : array
        output data to process

    """

    data = np.roll(data, data.shape[0]//fraction)
    data = np.append(data, data[0])

    return data


def verbose_print(verbose, *args):
    if verbose:
        print(*args)


def set_axis_aspect(ax, ratio=1):
    """
    sets the aspect ratio of a figure

    Parameters
    ----------
    ax : object
        figure axis to modify
    ratio  : float, optional
        sets the aspect ratio of the figure

    """

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


def colorbar(axes, plot,
             location='right', size=10,
             pad=0.05, num_format='%.1e',
             ticks = True,
             label = False):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    plot : matplotlib plot
        Plot being references for the scalebar
    location : str, optional
        position to place the colorbar
    size : int, optional
        percent size of colorbar relative to the plot
    pad : float, optional
        gap between colorbar and plot
    num_format : str, optional
        string format for the labels on colorbar
    label : str, optional
        sets the label for the axis
    """

    # Adds the scalebar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(location, size='{0}%'.format(size), pad=pad)
    cbar = plt.colorbar(plot, cax=cax, format=num_format)

    if not ticks:
        cbar.set_ticks([])

    if isinstance(label, str):
        cbar.set_label(label, rotation=270, labelpad= 15)


def find_nearest(array, value, averaging_number):
    """
    returns the indices nearest to a value in an image

    Parameters
    ----------
    array : float, array
        image to find the index closest to a value
    value : float
        value to find points near
    averaging_number : int
        number of points to find

    """
    idx = (np.abs(array-value)).argsort()[0:averaging_number]
    return idx


def loop_fitting_function(v, a1, a2, a3,
                          b1, b2, b3,
                          b4, b5, b6,
                          b7, b8,
                          Au, Al):
    """
    computes the loop fitting

    Parameters
    ----------
    V : float, array
        voltage array

    Return
    ----------
    {} : dict
        Branch1 : float, array
            top branch
        Branch2 : float, array
            bottom branch

    """

    S1 = ((b1+b2)/2) + ((b2-b1)/2)*erf((v-b7)/b5)
    S2 = ((b4+b3)/2) + ((b3-b4)/2)*erf((v-b8)/b6)
    Branch1 = (a1+a2)/2 + ((a2-a1)/2)*erf((v-Au)/S1)+a3*v
    Branch2 = (a1+a2)/2 + ((a2-a1)/2)*erf((v-Al)/S2)+a3*v

    return {'Branch1': Branch1, 'Branch2': Branch2}
