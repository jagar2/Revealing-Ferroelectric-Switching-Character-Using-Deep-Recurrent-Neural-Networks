import numpy as np
import os
from os.path import join as pjoin
import string
from matplotlib import (pyplot as plt, animation, colors,
                        ticker, path, patches, patheffects)
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    import moviepy as mpy
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
except:
    pass

import glob
from scipy import ndimage
from sklearn import (decomposition, preprocessing as pre, cluster, neighbors)
from scipy.signal import savgol_filter as sg
import keras
from keras.models import Sequential, Input, Model
from keras.layers import (Dense, Conv1D, Convolution2D, GRU, LSTM, Recurrent, Bidirectional, TimeDistributed,
                          Dropout, Flatten, RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
from keras.layers.core import Lambda
from keras.optimizers import Adam
from keras.regularizers import l1
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from scipy import special
from scipy.misc import imresize
from matplotlib.ticker import FormatStrFormatter
import datetime
from natsort import natsorted, ns
from skimage.transform import resize
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm


# Defines the colorlist
cmap = plt.get_cmap('viridis')
Path = path.Path
PathPatch = patches.PathPatch
erf = special.erf

# Defines a set of custom colormaps
cmap_2 = colors.ListedColormap(['#003057',
                                '#FFBD17'])
cmap_3 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02'])
cmap_4 = colors.ListedColormap(['#f781bf', '#e41a1c',
                                '#4daf4a', '#003057'])
cmap_5 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02',
                                '#7570b3', '#e7298a'])
cmap_6 = colors.ListedColormap(['#003057', '#1b9e77', '#d95f02',
                                '#7570b3', '#e7298a', '#66a61e'])
cmap_9 = colors.ListedColormap(['#e41a1c',  '#f781bf', '#003057',
                                '#a65628', '#984ea3', '#377eb8',
                                '#f46d43', '#cab2d6', '#4daf4a'])

# builds custom colormap for phase field model
color1 = plt.cm.viridis(np.linspace(.5, 1, 128))
color2 = plt.cm.plasma(np.linspace(1, .5, 128))
mymap = LinearSegmentedColormap.from_list(
    'my_colormap', np.vstack((color1, color2)))


def max_min_filter(data, ranges):
    """
    includes only data within a range of values as selected by the user.\n

    Parameters
    ----------
    data : numpy array
        array of loops
    ranges : array
        range of values to include

    Returns
    -------
    data : numpy array
        arary of loops
    """
    # checks if data is 3 dimensions
    if data.ndim == 3:
        # manually removes values which are too high or too low
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                # finds low and high values
                low = data[i, j] < min(ranges)
                high = data[i, j] > max(ranges)
                outliers = np.where(low + high)
                # removes found values and sets = nan
                data[i, j, outliers] = np.nan
    else:
        raise ValueError('Input data does not have a valid dimension')
    return data


def clean_and_interpolate(data, fit_type='spline'):
    """
    Function which removes bad datapoints

    Parameters
    ----------
    data : numpy, float
        data to clean
    fit_type : string  (optional)
        sets the type of fitting to use

    Returns
    -------
    data : numpy, float
        cleaned data
    """

    # sets all non finite values to nan
    data[~np.isfinite(data)] = np.nan
    # function to interpolate missing points
    data = interpolate_missing_points(data, fit_type)
    # reshapes data to a consistant size
    data = data.reshape(-1, data.shape[2])
    return data


def interpolate_missing_points(data, fit_type='spline'):
    """
    Interpolates bad pixels in piezoelectric hystereis loops.\n
    The interpolation of missing points alows for machine learning operations

    Parameters
    ----------
    data : numpy array
        array of loops
    fit_type : string (optional)
        selection of type of function for interpolation

    Returns
    -------
    data_cleaned : numpy array
        arary of loops
    """

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                if any(~np.isfinite(data[i, j, :, k])):

                    # selects the index where values are nan
                    ind = np.where(np.isnan(data[i, j, :, k]))

                    # if the first value is 0 copies the second value
                    if 0 in np.asarray(ind):
                        data[i, j, 0, k] = data[i, j, 1, k]

                    # selects the values that are not nan
                    true_ind = np.where(~np.isnan(data[i, j, :, k]))

                    # for a spline fit
                    if fit_type == 'spline':
                        # does spline interpolation
                        spline = interpolate.InterpolatedUnivariateSpline(point_values[true_ind],
                                                                          data[i, j, true_ind, k].squeeze())
                        data[i, j, ind, k] = spline(point_values[ind])

                    # for a linear fit
                    elif fit_type == 'linear':

                        # does linear interpolation
                        data[i, j, :, k] = np.interp(point_values,
                                                     point_values[true_ind],
                                                     data[i, j, true_ind, k].squeeze())

    return data.squeeze()


def make_folder(folder, **kwargs):
    """
    Function that makes new folders

    Parameters
    ----------

    folder : string
        folder where to save


    Returns
    -------
    folder : string
        folder where to save

    """

    if folder[0] != '.':
        folder = pjoin('./', folder)
    else:
        # Makes folder
        os.makedirs(folder, exist_ok=True)

    return (folder)


def labelfigs(axes, number, style='wb', loc='br', string_add='', size=14, text_pos='center'):
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
        sets the fontsize for the label
    text_pos : str, optional
        set the justification of the label
    """

    # Sets up various color options
    formating_key = {'wb': dict(color='w',
                                linewidth=1.5),
                     'b': dict(color='k',
                               linewidth=0),
                     'w': dict(color='w',
                               linewidth=0)}

    # Stores the selected option
    formatting = formating_key[style]

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

    # adds a cusom string
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


def add_scalebar_to_figure(axes, image_size, scale_size, units='nm', loc='br'):
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
    x_size, y_size = np.abs(np.floor(x_lim[1] - x_lim[0])),np.abs(np.floor(y_lim[1] - y_lim[0]))
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

def savefig(filename, printing):
    """
    Saves figure

    Parameters
    ----------
    filename : str
        path to save file
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    """

    # Saves figures at EPS
    if printing['EPS']:
        plt.savefig(filename + '.eps', format='eps',
                    dpi=printing['dpi'], bbox_inches='tight')

    # Saves figures as PNG
    if printing['PNG']:
        plt.savefig(filename + '.png', format='png',
                    dpi=printing['dpi'], bbox_inches='tight')

def custom_plt_format():

    """
    Loads the custom plotting format
    """

    # Loads the custom style
    plt.style.use('./custom.mplstyle')


def prints_all_BE_images(data, signal_clim,  plot_format,
                                            printing, folder_=''):

    """
    Function which prints all of the BE images

    Parameters
    ----------
    data : numpy, float
        raw data to plot
    signal_clim  : dictonary
        Instructions for extracting the data and plotting the data
    plot_format  : dictonary
        sets the format for the plots
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        path to save the files
    """

    # Graphs and prints all figures
    for (signal_name, signal), colorscale in signal_clim.items():

        # makes the folder to save the files
        folder = make_folder(folder_ + '/{}'.format(signal_name))

        # Cycles around each loop
        for cycle in (1, 2):

            # Bulids data name
            field = 'Out{0}{1}_mixed'.format(signal, cycle)

            # Displays loop status
            print('{0} {1}'.format(signal_name, cycle))

            # Loops around each voltage step
            for i in range(data[field].shape[2]):

                # Defines the figure and axes
                fig, ax1 = plt.subplots(figsize=(3, 3))

                #crops and rotates the image
                image, scale_factor = rotate_and_crop(data[field][:, :, i], plot_format['angle'],
                                                                                                        plot_format['frac_rm'])

                # Plots the data
                im = ax1.imshow(image, cmap='viridis')

                # Formats the figure
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
                ax1.set_xticks(np.linspace(0, image.shape[0], 5))
                ax1.set_yticks(np.linspace(0, image.shape[0], 5))
                im.set_clim(colorscale)
                ax1.set_facecolor((.55, .55, .55))

                # adds the scalebar to the images
                if plot_format['scalebar'] is not None:
                    add_scalebar_to_figure(ax1, plot_format['scalebar'][0]*scale_factor,
                                                                    plot_format['scalebar'][1])

                # Generates the filename
                filename = '{0}{1}_{2:03d}'.format(signal, cycle, i)

                # Saves the figure
                savefig(folder + '/' + filename,
                                printing)

                # Closes the figure
                plt.close(fig)

def plot_raw_BE_data(x, y, cycle, data, signals, printing, folder, cmaps = 'inferno'):

    """
    Plots the spectral curves from raw BE Data.

    Parameters
    ----------
    x : int
        x index to plot
    y : int
        y index to plot
    cycle : int
        cycle to plot
    data : dictonary
        raw Band Excitation Data to plot
    signals : list
        description of what to plot
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        folder where to save
    cmaps : string, optional
        colormap to use for plot
    """

    # Sets the colormap
    mymap = plt.get_cmap(cmaps)

    # Defines the axes positions
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # size of the voltage matrix
    v_size = data['Voltagedata_mixed'].shape[0]

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signals.items()):

        # Formats the figures
        axes[i].set_xlabel('Voltage (V)')
        axes[i].set_aspect('auto')
        axes[i].set_xticks(np.linspace(-1*(max(data['Voltagedata_mixed'])//5*5),
                                    max(data['Voltagedata_mixed'])//5*5, 7))
        axes[i].set_ylabel('{0} {1}'.format(signal, values['units']))
        axes[i].yaxis.set_major_formatter(
            ticker.FormatStrFormatter('{}'.format(values['format_str'])))

        # Sets d y-scales
        if np.isfinite(values['y_lim']).any():
            axes[i].set_ylim(values['y_lim'])
            axes[i].set_yticks(values['y_tick'])

        # Constructs the name and removes infinite values
        field = 'Out{0}{1}_mixed'.format(values['symbol'], cycle)

        # Stores values for graphing
        if signal in ['Amplitude', 'Resonance', 'Quality Factor']:

            finite_values = np.isfinite(data[field][x, y, :])
            signal_plot = data[field][x, y, finite_values]
            voltage_plot = data['VoltageDC_mixed'][finite_values, 1]

        elif signal == 'Phase':

            # Normalizes the phase data
            finite_values = np.isfinite(data[field][x, y, :])
            phi_data = data[field][x, y, finite_values]
            signal_plot = phi_data - \
                np.min(phi_data) - (np.max(phi_data) - np.min(phi_data)) / 2

        elif signal == 'Piezoresponse':

            # Computes and shapes the matrix for the piezorespons
            voltage_plot = np.concatenate([data['Voltagedata_mixed'][np.int(v_size / 4 * 3)::],
                                           data['Voltagedata_mixed'][0:np.int(v_size / 4 * 3)]])
            piezoresponse_data = data['Loopdata_mixed'][x, y, :]
            signal_plot = piezoresponse_data - np.mean(piezoresponse_data)
            signal_plot = np.concatenate([signal_plot[np.int(v_size / 4 * 3)::],
                                          signal_plot[0:np.int(v_size / 4 * 3)]])

        # plots the graph
        im = axes[i].plot(voltage_plot, signal_plot, '-ok', markersize=3)

        # Removes Y-label for Arb. U.
        if signal in ['Amplitude', 'Piezoresponse']:
            axes[i].set_yticklabels('')

        # Adds markers to plot
        for index in [0, 14, 24, 59, 72]:
            axes[i].plot(voltage_plot[index], signal_plot[index],
                         'ok', markersize=8,
                         color=mymap((data['VoltageDC_mixed'][index, 1] + 16) / 32))

        # labels figures
        labelfigs(axes[i], i)

        plt.tight_layout()

        # Prints the images
        filename = 'Raw_Loops_{0:d}_{1:d}'.format(x, y)
        savefig(folder + '/' + filename,
                                printing )

def plot_raw_BE_images_for_movie(data, signals_clim,
                                                            plot_format, printing,
                                                            folder = 'Movie Images',):

    """
    Plots raw BE data

    Parameters
    ----------
    data : dictonary
        imported collection of data
    signals_clim : list
        description of what to plot
    plot_format : dictonary
        sets the format for what to plot
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string, optional
        folder where to save
    """

    # makes a copy of the voltage and reshapes it
    voltage = np.copy(data['data']['Voltagedata_mixed'])
    voltage_steps = voltage.shape[0]
    voltage = roll_and_append(voltage)[::-1]
    voltage = np.append(voltage,voltage)

    # Cycles around each loop
    for cycle in (1, 2):

            # Loops around each voltage step
            for i in range(voltage_steps):

                # Defines the axes positions
                fig = plt.figure(figsize=(8,12))
                ax1 = plt.subplot2grid((3,2), (0,0))
                ax2 = plt.subplot2grid((3,2), (0,1))
                ax3 = plt.subplot2grid((3,2), (1, 0))
                ax4 = plt.subplot2grid((3,2), (1, 1))
                ax5 = plt.subplot2grid((3,2), (2, 0), colspan=2)
                axes = (ax1, ax2, ax3, ax4)

                # Sets the axes labels and scales
                for ax in axes:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks(np.linspace(0, 1024, 5))
                    ax.set_yticks(np.linspace(0, 1024, 5))
                    ax.set_facecolor((.55, .55, .55))


                # Plots the response maps
                for j, (signals, colorscale) in enumerate(signals_clim.items()):

                    (signal_name, signal, formspec) = signals

                    # Defines the data location
                    field = 'Out{0}{1}_mixed'.format(signal, cycle)

                    # Plots and formats the images
                    image, scale_frac = rotate_and_crop(data['data'][field][:,:,i],plot_format['angle'],
                                                plot_format['frac_rm'])
                    im=axes[j].imshow(image)
                    axes[j].set_title(signal_name)
                    im.set_clim(colorscale)

                    # Sets the colorbars
                    divider = make_axes_locatable(axes[j])
                    cax = divider.append_axes('right', size='10%', pad=0.05)
                    cbar = plt.colorbar(im, cax=cax, format=formspec)

                # Plots the voltage graph
                im5 = ax5.plot(voltage, 'ok')
                ax5.plot(i + (cycle - 1) * voltage_steps,
                         voltage[i],
                         'rs', markersize=12)

                # Generates the filename
                filename = 'M{0}_{1:03d}'.format(cycle, i)

                # saves the figure
                savefig(folder + '/' + filename,
                                printing)

                # Closes the figure
                plt.close(fig)

def make_movie(movie_name, input_folder, output_folder, file_format,
                            fps, output_format = 'mp4', reverse = False):

    """
    Function which makes movies from an imageseries

    Parameters
    ----------
    movie_name : string
        name of the movie
    input_folder  : string
        folder where the image series is located
    output_folder  : string
        folder where the movie will be saved
    file_format  : string
        sets the format of the files to import
    fps  : numpy, int
        frames per second
    output_format  : string, optional
        sets the format for the output file
        supported types .mp4 and gif
        animated gif create large files
    reverse : bool, optional
        sets if the movie will be one way of there and back
    """

    # searches the folder and finds the files
    file_list = glob.glob('./' + input_folder + '/*.' + file_format)

    # Sorts the files by number makes 2 lists to go forward and back
    list.sort(file_list)
    file_list_rev = glob.glob('./' + input_folder + '/*.' + file_format)
    list.sort(file_list_rev,reverse=True)

    # combines the filelist if including the reverse
    if reverse:
        new_list = file_list + file_list_rev
    else:
        new_list = file_list


    if output_format == 'gif':
        # makes an animated gif from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_gif(output_folder + '/{}.gif'.format(movie_name), fps=fps)
    else:
        # makes and mp4 from the images
        clip = ImageSequenceClip(new_list, fps=fps)
        clip.write_videofile(output_folder + '/{}.mp4'.format(movie_name), fps=fps)

def rotate_and_crop(image_, angle=60.46, frac_rm = 0.17765042979942694):

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
        image which is rotated and croped
    scale_factor : float
        scaling factor for the image following rotation
    """
    # makes a copy of the image
    image = np.copy(image_)

    # replaces all points with the minimum value
    image[~np.isfinite(image)] = np.nanmin(image)

    # rotates the image
    rot_topo = ndimage.interpolation.rotate(image,90-angle,cval=np.nanmin(image))

    # crops the image
    pix_rem = int(rot_topo.shape[0]*(frac_rm))
    crop_image = rot_topo[pix_rem:rot_topo.shape[0]-pix_rem,pix_rem:rot_topo.shape[0]-pix_rem]

    # returns the scale factor for the new image size
    scale_factor =  (np.cos(np.deg2rad(angle))+np.cos(np.deg2rad(90-angle)))*(1-frac_rm)

    return crop_image, scale_factor

def plot_loopfit_results(data, signal_clim, printing, folder, plot_format):

    """
    Saves figure

    Parameters
    ----------
    data : dictonary
        dictionary containing the loopfitting results
    signal_clim : list
        description of what to plot
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string
        folder where the images will be saved
    plot_format : dictonary
        sets the format for what to plot
    """

    # Defines the figure and axes
    fig, axes = plt.subplots(5, 6, figsize=(18,15))
    axes = axes.reshape(30)

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signal_clim.items()):

        # Sets the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0,59,5))
        axes[i].set_yticks(np.linspace(0,59,5))
        axes[i].set_title('{0}'.format(values['label']))
        axes[i].set_facecolor((.55, .55, .55))

        field = '{}'.format(values['data_loc'])

        # rotates the image
        #if plot_format['rotation']:
        #    image, scalefactor  = rotate_and_crop(data[field], plot_format['angle'],
        #        plot_format['frac_rm'])
        #else:
        #    scalefactor = 1

        # Plots the graphs either abs of values or normal
        if i in {13,20,21,22,23}:

            # plots the image map
            im = plot_imagemap(axes[i], np.abs(data[field]), plot_format, clim = values['c_lim'])

        else:

            im = plot_imagemap(axes[i], data[field], plot_format, clim = values['c_lim'])

        # labels figures
        labelfigs(axes[i],i)

    # Deletes unused figures
    fig.delaxes(axes[28])
    fig.delaxes(axes[29])

    # Saves Figure
    plt.tight_layout(pad=0, h_pad=-20)
    savefig(folder + '/loopfitting_results', printing)

def conduct_PCA(loops, n_components=15, verbose=True):
    """
    Computes the PCA and forms a low-rank representation of a series of response curves
    This code can be applied to all forms of response curves.
    loops = [number of samples, response spectra for each sample]

    Parameters
    ----------
    loops : numpy array
        1 or 2d numpy array - [number of samples, response spectra for each sample]
    n_components : int, optional
        int - sets the number of componets to save
    verbose : bool, optional
        output operational comments

    Returns
    -------
    PCA : object
        results from the PCA
    PCA_reconstructed : numpy array
        low-rank representation of the raw data reconstructed based on PCA denoising
    """

    # resizes the array for hyperspectral data
    if loops.ndim == 3:

        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0}x {1}]'.format(
            loops.shape[0], loops.shape[1]))

    elif loops.ndim == 2:
        pass
    else:
        raise ValueError("data is of an incorrect size")

    if np.isnan(loops).any():
        raise ValueError(
            'data has infinite values consider using a imputer \n see interpolate_missing_points function')

    # Sets the number of components to save
    pca = decomposition.PCA(n_components=n_components)

    # Computes the PCA of the piezoelectric hysteresis loops
    PCA = pca.fit(loops)

    # does the inverse transform - creates a low rank representation of the data
    # this process denoises the data
    PCA_reconstructed = pca.inverse_transform(pca.transform(loops))

    # resized the array for hyperspectral data
    try:
        PCA_reconstructed = PCA_reconstructed.reshape(original_size, original_size, -1)
    except:
        pass

    return PCA, PCA_reconstructed


def plot_pca_results(pca, data, signal_info, printing, folder, plot_format, signal,
                                 verbose=False, letter_labels=True, filename='./PCA_maps',
                                num_of_plots=True):
    """
    Plots the results from the computed principal component analysis

    Parameters
    ----------
    PCA : object
        results from the PCA
    data : dictionary
        complete raw band excitation piezoresponse data
    signal_info : dictonary
        contains information about the signal used for plotting
    printing : dictonary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string
        folder where the images will be saved
    plot_format : dictonary
        sets the format for what to plot
    signal : string
        sets the name of what to plot
    verbose : bool, optional
        sets if displays information
    letter_labels : bool, optional
        sets if to add figure labels
    filename : bool, optional
        sets filename where to save the images
    num_of_plots : int, optional
        sets a number of plots or components to show

    """
    # extracts the data to plot
    voltage = data['raw']['voltage']
    loops = data['sg_filtered'][signal]
    min_ = np.min(pca.components_.reshape(-1))
    max_ = np.max(pca.components_.reshape(-1))
    count = 0

    # selects number of plots to display if not given by used
    if num_of_plots == True:
        num_of_plots = pca.n_components_

    # stores the number of plots in a row
    mod = num_of_plots//(np.sqrt(num_of_plots)//1).astype(int)

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots*2, mod = mod)

    # resizes the array for hyperspectral data
    if loops.ndim == 3:

        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])

        # prints that the data has been resized
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))

    elif loops.ndim == 2:
        original_size = np.sqrt(loops.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    # computes the PCA maps
    PCA_maps = pca_weights_as_embeddings(pca, loops, num_of_components=num_of_plots)

    # Formats figures
    for i, ax in enumerate(ax):

        # Checks if axes is an image or a plot
        if (i // mod % 2 == 0):
            pc_number = i - mod * (i // (mod*2))
            im = plot_imagemap(ax, PCA_maps[:, pc_number], plot_format)

            # labels figures
            labelfigs(ax, i, string_add='PC {0:d}'.format(pc_number+1), loc='bm')

        else:

            # Plots the PCA egienvector and formats the axes
            ax.plot(voltage, pca.components_[
                    i - mod - ((i // mod) // 2) * mod], 'k')

            # Formats and labels the axes
            ax.set_xlabel('Voltage')
            ax.set_ylabel(signal_info[signal]['units'])
            ax.set_yticklabels('')
            ax.set_ylim([min_,max_])
            if signal_info[signal]['pca_range'] is not None:
                ax.set_ylim(signal_info[signal]['pca_range'])

        # labels figures
        if letter_labels:
            if (i // mod % 2 == 0):
                labelfigs(ax, count)
                count += 1

        # sets the aspect ratio of the figure
        set_axis_aspect(ax)

    plt.tight_layout(pad=0, h_pad=0)

    savefig(folder + '/' +filename, printing)

def verbose_print(verbose, *args):

    """
    Function which prints some information

    Parameters
    ----------

    verbose : bool, optional
        sets if displays information

    """
    if verbose:
        print(*args)

def layout_graphs_of_arb_number(graph, mod=None):
    """
    Sets the layout of graphs in matplotlib in a pretty way based on the number of plots

    Parameters
    ----------
    graphs : int
        number of axes to make

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """

    if mod == None:
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


def pca_weights_as_embeddings(pca, loops, num_of_components=0, verbose=True):
    """
    Computes the eigenvalue maps computed from PCA

    Parameters
    ----------
    pca : object
        computed PCA
    loops: numpy array
        raw piezoresponse data
    num_of _components: int
        number of PCA components to compute

    Returns
    -------
    fig : matplotlib figure
        handel to figure being created.
    axes : numpy array (axes)
        numpy array of axes that are created.
    """
    if loops.ndim == 3:
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))

    if num_of_components == 0:
        num_of_components = pca.n_components_

    PCA_embedding = pca.transform(loops)[:, 0:num_of_components]

    return (PCA_embedding)


def plot_imagemap(ax, data, plot_format, clim = None):

        """
        Plots an imagemap

        Parameters
        ----------
        axis : matplotlib, object
            axis which is plotted
        data  : numpy, float
            data to plot
        clim  : numpy, float, optional
            sets the climit for the image
        color_bar  : bool, optional
            selects to plot the colorbar bar for the image
        """
        if data.ndim == 1:
            data = data.reshape(np.sqrt(data.shape[0]).astype(int), np.sqrt(data.shape[0]).astype(int))

        if plot_format['color_map'] is None:
            cmap = plt.get_cmap('viridis')
        else:
            cmap = plt.get_cmap(plot_format['color_map'])

        if plot_format['rotation']:
            data, scalefactor  = rotate_and_crop(data, angle=plot_format['angle'],
                                                        frac_rm = plot_format['frac_rm'])
        else:
            scalefactor = 1

        if clim is None:
            im = ax.imshow(data,  cmap=cmap)
        else:
            im = ax.imshow(data, clim = clim, cmap=cmap)

        ax.set_yticklabels('')
        ax.set_xticklabels('')

        if plot_format['color_bars']:
            # Adds the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1e')

        if plot_format['add_scalebar'] is not False:
            add_scalebar_to_figure(ax, plot_format['scalebar'] [0]*scalefactor, plot_format['scalebar'] [1])

        return im

def set_axis_aspect(ax,ratio = 1):

        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

def sg_filter_data(data_, num_to_remove=3, window_length=7, polyorder=3, fit_type='spline'):
    """
    Applies a Savitzky-Golay filter to the data which is used to remove outlier or noisy points from the data

    Parameters
    ----------
    data : numpy, array
        array of loops
    num_to_remove : numpy, int
        sets the number of points to remove
    window_length : numpy, int
        sets the size of the window for the sg filter
    polyorder : numpy, int
        sets the order of the sg filter
    fit_type : string
        selection of type of function for interpolation

    Returns
    -------
    cleaned_data : numpy array
        array of loops
    """
    data = np.copy(data_)

    # reshapes the data such that it can run with different data sizes
    if data.ndim == 2:
        data = data.reshape(np.sqrt(data.shape[0]).astype(int),
                            np.sqrt(data.shape[0]).astype(int), -1)
        data = np.expand_dims(data, axis=3)
    elif data.ndim == 3:
        data = np.expand_dims(data, axis=3)

    cleaned_data = np.copy(data)

    # creates a vector of the size of the data
    point_values = np.linspace(0, 1, data.shape[2])

    # Loops around the x index
    for i in range(data.shape[0]):

        # Loops around the y index
        for j in range(data.shape[1]):

            # Loops around the number of cycles
            for k in range(data.shape[3]):

                sg_ = sg(data[i, j, :, k],
                         window_length=window_length, polyorder=polyorder)
                diff = np.abs(data[i, j, :, k] - sg_)
                sort_ind = np.argsort(diff)
                remove = sort_ind[-1 * num_to_remove::].astype(int)
                cleaned_data[i, j, remove, k] = np.nan

    cleaned_data = clean_and_interpolate(cleaned_data, fit_type)

    return cleaned_data

def compute_nmf(model, data):

    # Fits the NMF model
    W = model.fit_transform(np.rollaxis(data - np.min(data),1))
    H = np.rollaxis(model.components_,1)

    return W, H

def plot_NMF_maps(voltage, nmf,
                                printing, plot_format,
                                signal_info,
                                folder='./',
                                verbose=False, letter_labels=False,
                                custom_order = None):

    W = nmf[0]
    H = nmf[1]

    # extracts the number of maps
    num_of_plots = H.shape[1]

    image_size = np.sqrt(H.shape[0]).astype(int)

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots*2, mod = num_of_plots)

    min_ = np.min(W[:,:].reshape(-1))
    max_ = np.max(W[:,:].reshape(-1))

    if custom_order is not None:
        order=custom_order

    for i, ax in enumerate(ax):


        # Checks if axes is an image or a plot
        if (i // num_of_plots % 2 == 0):
            # converts axis number to index number
            k = i - ((i // num_of_plots) // 2) * num_of_plots

            im = plot_imagemap(ax, H[:, order[i]].reshape(image_size,image_size),plot_format)

            # labels figures
            if letter_labels:
                if (i // num_of_plots % 2 == 0):
                    labelfigs(ax, k)
        else:
            # converts axis number to index number
            k = i - num_of_plots - ((i // num_of_plots) // 2) * num_of_plots

            ax.plot(voltage, W[:,order[k]],'k')

            ax.set_xlabel('Voltage')
            ax.set_ylim([min_,max_])
            ax.set_ylabel(signal_info['units'])
            ax.set_yticklabels('')

            set_axis_aspect(ax)


    plt.tight_layout(pad=0)

    savefig(folder + '/nmf_' + signal_info['symbol'] , printing)

def normalize_data(data, data_normal = None):

    if data_normal is None:
        data_norm = np.copy(data)
        data_norm -= np.mean(data_norm.reshape(-1))
        data_norm /= np.std(data_norm)
    else:
        data_norm = np.copy(data)
        data_norm -= np.mean(data_normal.reshape(-1))
        data_norm /= np.std(data_normal)

    return data_norm


def cluster_loops(input_data, channel, clustering, seed=[], pca_in=True):
    """
    Clusters the loops

    Parameters
    ----------
    data : float
        data for clustering
    int_cluster : int
        first level divisive clustering
    c_cluster : int
        c level divissive clustering
    a_cluster : int
        a level divissive clustering
    seed : int, optional
        fixes the seed for replicating results

    """

    if pca_in is True:
        data = input_data['sg_filtered'][channel]
    else:
        data=pca_in

    data_piezo = input_data['sg_filtered']['piezoresponse']

    if seed!=[]:
        # Defines the random seed for consistant clustering
        np.random.seed(seed)

    # Scales the data for clustering
    scaled_data = pre.StandardScaler().fit_transform(data)
    scaled_data_piezo = pre.StandardScaler().fit_transform(data_piezo)

    # Kmeans clustering of the data into c and a domains
    cluster_ca = cluster.KMeans(
        n_clusters=clustering['initial_clusters']).fit_predict(scaled_data_piezo)

    # K-means clustering of a domains
    a_map = np.zeros(data.shape[0])
    a_cluster = cluster.KMeans(n_clusters=clustering['a_clusters']).fit_predict(
        scaled_data[cluster_ca == 1])
    a_map[cluster_ca == 1] = a_cluster + 1

    # Kmeans clustering of c domains
    c_map = np.zeros(data.shape[0])
    c_cluster = cluster.KMeans(n_clusters=clustering['c_clusters']).fit_predict(
        scaled_data[cluster_ca == 0])
    c_map[cluster_ca == 0] = c_cluster + clustering['a_clusters'] + 1

    # Enumerates the k-means clustering map for ploting
    combined_map = a_map + c_map

    return(combined_map, cluster_ca, c_map, a_map)


def plot_hierachical_cluster_maps(cluster_results, names, plot_format):

    combined_map, cluster_ca, c_map, a_map =cluster_results

    # Defines the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Loops around all the clusteres found
    for i, name in enumerate(names):

        (title, cluster_map) = name

        # sets the order of the plots
        if cluster_map == 'cluster_ca':
            i = 0
        elif cluster_map == 'a_map':
            i = 2
        elif cluster_map == 'c_map':
            i = 1

        size_image = np.sqrt(c_map.shape[0]).astype(int)-1

        num_colors = len(np.unique(eval(cluster_map)))
        scales = [np.max(eval(cluster_map))-(num_colors -.5), np.max(eval(cluster_map))+.5]

        # Formats the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0, size_image, 5))
        axes[i].set_yticks(np.linspace(0, size_image, 5))
        axes[i].set_title(title)
        axes[i].set_facecolor((.55, .55, .55))

        if plot_format['rotation']:
            image, scalefactor  = rotate_and_crop(eval(cluster_map).reshape(size_image+1, size_image+1),
                                                angle=plot_format['angle'],
                                                frac_rm = plot_format['frac_rm'])
        else:
            scalefactor = 1
            image = eval(cluster_map).reshape(size_image+1, size_image+1)

        # Plots the axes
        im = axes[i].imshow(image,
                            cmap=eval(f'cmap_{num_colors}'), clim=scales)

        labelfigs(axes[i], i, loc='br')
        add_scalebar_to_figure(axes[i], plot_format['scalebar'][0]*scalefactor,
                                                        plot_format['scalebar'][1], loc='br')

        # Formats the colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%d')
        cbar.set_ticks([])

def plot_clustered_hysteresis(voltage,
                                hysteresis,
                                cluster_results,
                                plot_format,
                                signal_info,
                                channel,
                                printing,
                                folder):

    """
    Plots the cluster maps and the average hysteresis loops

    Parameters
    ----------
    voltage : numpy array, float
        voltage vector
    Piezoresponse  : numpy array, float
        Piezoresponse data from the loops
    combined_cluster_map  : numpy array, int
        computed cluster map of the data
    loop_ylim: numpy array, float
        sets the y limit for the hysteresis loops
    loops_xlim: numpy array, float
        sets the x limit for the hysteresis loops

    """

    combined_map, cluster_ca, c_map, a_map =cluster_results

    # organization of the raw data

    num_pix = np.sqrt(combined_map.shape[0]).astype(int)
    num_clusters = len(np.unique(combined_map))

    # Preallocates some matrix
    clustered_maps= np.zeros((num_clusters, num_pix, num_pix))
    clustered_ave_hysteresis = np.zeros((num_clusters, hysteresis.shape[1]))

    cmap_=eval(f'cmap_{num_clusters}')

    # Loops around the clusters found
    for i in range(num_clusters):

        # Stores the binary maps
        binary = (combined_map == i + 1)
        clustered_maps[i, :, :] = binary.reshape(num_pix, num_pix)

        # Stores the average piezoelectric loops
        clustered_ave_hysteresis[i] = np.mean(
            hysteresis[binary], axis=0)

    fig, ax = layout_graphs_of_arb_number(num_clusters + 1)

    for i in range(num_clusters + 1):

        if i == 0:

            scales = [np.max(combined_map)-(num_clusters -.5),
                        np.max(combined_map)+.5]

            # Formats the axes
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_xticks(np.linspace(0, num_pix, 5))
            ax[i].set_yticks(np.linspace(0, num_pix, 5))
            ax[i].set_facecolor((.55, .55, .55))

            if plot_format['rotation']:
                image, scalefactor  = rotate_and_crop(combined_map.reshape(num_pix, num_pix),
                                                    angle=plot_format['angle'], frac_rm = plot_format['frac_rm'])
            else:
                scalefactor = 1
                image = combined_map.reshape(num_pix, num_pix)

            labelfigs(ax[i], i, loc='tr')
            add_scalebar_to_figure(ax[i],  plot_format['scalebar'][0]*scalefactor,
                                                            plot_format['scalebar'][1], loc='tr')

            # Plots the axes
            im = ax[i].imshow(image,
                                cmap=cmap_, clim=scales)

            add_colorbar(ax[i], im, ticks=False)

        else:

            # Plots the graphs
            hys_loop = ax[i].plot(voltage, clustered_ave_hysteresis[i-1],cmap_.colors[i-1])

            # formats the axes
            ax[i].yaxis.tick_left()
            ax[i].set_xticks(signal_info['voltage']['x_tick'])
            ax[i].yaxis.set_label_position('left')
            ax[i].set_ylabel(signal_info['piezoresponse']['units'])
            ax[i].set_xlabel(signal_info['voltage']['units'])
            ax[i].yaxis.get_major_formatter().set_powerlimits((0, 1))
            if signal_info['piezoresponse']['y_lim'] != None:
                ax[i].set_ylim(signal_info['piezoresponse']['y_lim'])
            ax[i].set_yticklabels([])

            pos = ax[i].get_position()

            # Posititions the binary image
            axes_in = plt.axes([pos.x0+.06,
                                pos.y0+.025,
                                .1, .1])
            combined_map.reshape(num_pix, num_pix)

            if plot_format['angle']:
                imageb, scalefactor  = rotate_and_crop(clustered_maps[i-1, :, :],
                                                    angle=plot_format['angle'], frac_rm = plot_format['frac_rm'])
            else:
                scalefactor = 1
                imageb = clustered_maps[i-1, :, :]

            # Plots and formats the binary image
            axes_in.imshow(imageb,
                           cmap=cmap_2)
            axes_in.tick_params(axis='both', labelbottom=False, labelleft=False)

            labelfigs(ax[i], i, loc='br')

            set_axis_aspect(ax[i])

            savefig(folder + '/'  + channel,
                        printing)


def add_colorbar(axes, plot, location='right', size=10, pad=0.05, format='%.1e', ticks = True):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    axes : matplotlib axes
        axes which to add the plot to
    axes : matplotlib plot
        Plot being references for the scalebar
    location : str, optional
        position to place the colorbar
    size : int, optional
        percent size of colorbar realitive to the plot
    pad : float, optional
        gap between colorbar and plot
    format : str, optional
        string format for the labels on colorbar
    """

    # Adds the scalebar
    divider = make_axes_locatable(axes)
    cax = divider.append_axes(location, size='{0}%'.format(size), pad=pad)
    cbar = plt.colorbar(plot, cax=cax, format=format)

    if not ticks:
        cbar.set_ticks([])

def visualize_cleaned_data(data, i, printing, folder='./'):

    fig, ax = layout_graphs_of_arb_number(5,5)
    voltage = data['raw']['voltage']
    count = 0


    #build a looping function to display all the cleaned data
    # Plots each of the graphs
    for j, (signal, values) in enumerate(data['raw'].items()):
        if j in [1,2,3,4,5]:
            ax[count].plot(voltage, data['interpolated'][signal][i],'k')
            ax[count].plot(voltage, data['sg_filtered'][signal][i],'r')

            # Formats the figures
            ax[count].set_xlabel('Voltage (V)')
            ax[count].set_aspect('auto')
            ax[count].set_xticks(np.linspace(-15, 15, 7))
            ax[count].set_ylabel(data['signal_info'][signal]['units'])
            ax[count].yaxis.set_major_formatter(
                FormatStrFormatter(data['signal_info'][signal]['format_str']))

            # Removes Y-label for Arb. U.
            if signal in ['amplitude', 'piezoresponse']:
                ax[count].set_yticklabels('')

            # labels figures
            labelfigs(ax[count], count)

            plt.tight_layout()

            count += 1

    # Prints the images
    filename = '/Raw_Loops_{0:d}'.format(i)
    savefig(folder + filename,
                printing)

def rnn_auto(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             n_step, lr = 3e-5, drop_frac=0.,
             bidirectional=True, l1_norm = 1e-4,
             batch_norm=[False, False],**kwargs):
    """
    Function which builds the reccurrent neural network autoencoder

    Parameters
    ----------
    layer : string; options: 'lstm','gru'
        selects the layer type
    size  : numpy, int
        sets the size of encoding and decoding layers in the network
    num_encode_layers  : numpy, int
        sets the number of encoding layers in the network
    num_decode_layers : numpy, int
        sets the number of decoding layers in the network
    embedding : numpy, int
        sets the size of the embedding layer
    n_steps : numpy, int
        length of the input time series
    lr : numpy, float
        sets the learning rate for the model
    drop_frac : numpy, float
        sets the dropout fraction
    bidirectional : numpy, bool
        selects if the model is linear or bidirectional
    l1_norm : numpy. float
        sets the lambda value of the l1 normalization. The larger the value the greater the
        sparsity. None can be passed to exclude the use or l1 normailzation.

    Returns
    -------
    model : Keras, object
        Keras tensorflow model
    """

    # Selects the type of RNN neurons to use
    if layer_type == 'lstm':
        layer = LSTM
    elif layer_type == 'gru':
        layer = GRU

    # defines the model
    model = Sequential()

    # selects if the model is bidirectional
    if bidirectional:
        wrapper = Bidirectional
        # builds the first layer

        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(num_encode_layers > 1)),
                        input_shape=(n_step, 1)))
        add_dropout(model, drop_frac)
    else:
        wrapper = lambda x: x
        # builds the first layer
        model.add(wrapper(layer(size, return_sequences=(num_encode_layers > 1),
                        input_shape=(n_step, 1))))
        add_dropout(model, drop_frac)


    # builds the encoding layers
    for i in range(1, num_encode_layers):
        model.add(wrapper(layer(size, return_sequences=(i < num_encode_layers - 1))))
        add_dropout(model, drop_frac)

    # adds batch normalization prior to embedding layer
    if batch_norm[0]:
        model.add(BatchNormalization())

    # builds the embedding layer
    if l1_norm == None:
        # embedding layer without l1 regulariization
        model.add(Dense(embedding, activation='relu', name='encoding'))
    else:
        # embedding layer with l1 regularization
        model.add(Dense(embedding, activation='relu', name='encoding',activity_regularizer=l1(l1_norm)))

    # adds batch normalization after embedding layer
    if batch_norm[1]:
        model.add(BatchNormalization())

    # builds the repeat vector
    model.add(RepeatVector(n_step))

    # builds the decoding layer
    for i in range(num_decode_layers):
        model.add(wrapper(layer(size, return_sequences=True)))
        add_dropout(model, drop_frac)

    # builds the time distributed layer to reconstruct the original input
    model.add(TimeDistributed(Dense(1, activation='linear')))

    # complies the model
    model.compile(Adam(lr), loss='mse')

    run_id = get_run_id(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             lr, drop_frac, bidirectional, l1_norm,
             batch_norm)

    # returns the model
    return model, run_id

def add_dropout(model, value):
    if value > 0:
        return model.add(Dropout(value))
    else:
        pass

def get_run_id(layer_type, size, num_encode_layers,
             num_decode_layers, embedding,
             lr, drop_frac,
             bidirectional, l1_norm,
             batch_norm, **kwargs):

    run = (f"{layer_type}_size{size:03d}_enc{num_encode_layers}_emb{embedding}_dec{num_decode_layers}_lr{lr:1.0e}"
           f"_drop{int(100 * drop_frac)}").replace('e-', 'm')
    if Bidirectional:
        run = 'Bidirect_' + run
    if layer_type == 'conv':
        run += f'_k{kernel_size}'
    if np.any(batch_norm):

        if batch_norm[0]:
            ind = 'T'
        else:
            ind = 'F'

        if batch_norm[1]:
            ind1 = 'T'
        else:
            ind1 = 'F'

        run += f'_batchnorm_{ind}{ind1}'
    return run

def get_activations(model, X=[], i=[], mode='test'):

    """
    support function to get the activations of a specific layer
    this function can take either a model and compute the activations or can load previously
    generated activations saved as an numpy array

    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    """

    if isinstance(model, str):
        activation = np.load(model)
        print(f'model {model} loaded from saved file')
    else:
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, model)

    return activation

def get_ith_layer_output(model, X, i, mode='test'):

    """
    Computes the activations of a specific layer
    see https://keras.io/getting-started/faq/#keras-faq-frequently-asked-keras-questions'


    Parameters
    ----------
    model : keras model, object
        pre-trained keras model
    X  : numpy array, float
        Input data
    i  : numpy, int
        index of the layer to extract
    mode : string, optional
        test or train, changes the model behavior to scale the network properly when using
        dropout or batchnorm
    """
    get_ith_layer = keras.backend.function(
        [model.layers[0].input, keras.backend.learning_phase()], [model.layers[i].output])
    layer_output = get_ith_layer([X, 0 if mode=='test' else 1])[0]
    return layer_output

def plot_embedding_maps(data, printing, plot_format, folder, verbose=False, letter_labels=False,
                         filename='./embedding_maps', num_of_plots=True, ranges=None):
    """
    Adds a colorbar to a imageplot

    Parameters
    ----------
    data : raw data to plot of embeddings
        data of embeddings
    add_colorbars : bool, optional
        adds colorbars to images
    verbose : bool, optional
        sets the verbosity level
    letter_labels : bool, optional
        adds letter labels for use in publications
    add_scalebar : bool, optional
        sets whether a scalebar is added to the maps
    filename : str, optional
        sets the path and filename for the exported images
    print_EPS : bool, optional
        to export as EPS
    print_PNG : bool, optional
        to export as PNG
    dpi : int, optional
        resolution of exported image
    num_of_plots : int, optional
            number of principle componets to show
    ranges : float, optional
            sets the clim of the images
    """
    if num_of_plots:
        num_of_plots = data.shape[data.ndim - 1]

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(num_of_plots, mod=4)
    # resizes the array for hyperspectral data

    if data.ndim == 3:
        original_size = data.shape[0].astype(int)
        data = data.reshape(-1, data.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            data.shape[0], data.shape[1]))
    elif data.ndim == 2:
        original_size = np.sqrt(data.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    for i in range(num_of_plots):
        if plot_format['rotation']:
            image, scalefactor  = rotate_and_crop(data[:, i].reshape(original_size, original_size),
                                                        angle=plot_format['angle'], frac_rm = plot_format['frac_rm'])
        else:
            image = data[:, i].reshape(original_size, original_size)
            scalefactor = 1
        im = ax[i].imshow(image)
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')

        if ranges is None:
            pass
        else:
            im.set_clim(0,ranges[i])

        if plot_format['color_bars']:
            add_colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='emb. {0}'.format(i + 1), loc='bm')

        if plot_format['add_scalebar'] is not False:
            add_scalebar_to_figure(ax[i], plot_format['scalebar'][0]*scalefactor,
            plot_format['scalebar'][1])

    plt.tight_layout(pad=1)

    savefig(folder + '/' + filename, printing)

    return(fig)


def generate_generator_images(model, encode, voltage, number, averaging_number,
                                                    ranges, folder, plot_format,  printing, graph_layout = [4,4]):

    # Defines the colorlist
    cmap = plt.get_cmap('viridis')

    ind = np.where(np.mean(encode,axis=0)>0)
    encode_small = encode[:,ind].squeeze()

    mean_loop = model.predict(np.atleast_2d(np.zeros((encode.shape[1])))).squeeze()

    for i in tqdm(range(number)):

        # builds the figure
        fig, ax = plt.subplots(graph_layout[0] // graph_layout[1]  + (graph_layout[0]  % graph_layout[1] > 0), graph_layout[1],
                             figsize=(3 * graph_layout[1], 3 * (graph_layout[0]  // graph_layout[1] + (graph_layout[0]  % graph_layout[1] > 0))))
        ax = ax.reshape(-1)

        for j in range(len(ranges)):

            value = np.linspace(0,ranges[j],number)

            if i == 0:
                gen_value = np.zeros((encode.shape[1]))
            else:
                idx = find_nearest(encode_small[:,j],value[i],averaging_number)
                gen_value = np.mean(encode[idx],axis=0)
                gen_value[j] = value[i]

            generated = model.predict(np.atleast_2d(gen_value)).squeeze()
            ax[j].plot(voltage, generated, color = cmap((i+1)/number))
            ax[j].set_ylim(-2,2)
            ax[j].set_yticklabels('')
            ax[j].plot(voltage, mean_loop, color = cmap((0+1)/number))
            ax[j].set_xlabel('Voltage (V)')

            pos = ax[j].get_position()

            # plots and formats the binary cluster map
            axes_in = plt.axes([pos.x0-.0105, pos.y0,.06*4, .06*4])
            if plot_format['rotation']:
                imageb, scalefactor  = rotate_and_crop(encode_small[:, j].reshape(60, 60),
                                                    angle=plot_format['angle'], frac_rm = plot_format['frac_rm'])
            else:
                scalefactor = 1
                imageb = encode_small[:, j].reshape(60, 60)
            axes_in.imshow(imageb, clim=[0,ranges[j]])
            axes_in.set_yticklabels('')
            axes_in.set_xticklabels('')

        ax[0].set_ylabel('Piezoresponse (Arb. U.)')


        savefig(pjoin(folder,f'{i:04d}_maps'), printing)
        plt.close(fig)

def find_nearest(array,value, averaging_number):
    idx = (np.abs(array-value)).argsort()[0:averaging_number]
    return idx

def plot_generative_piezoresponse_results(model,embeddings,voltage,
                                          ranges,number,average_number, printing, plot_format,
                                          folder, y_scale=[-1.6,1.6]):

    cmap = plt.cm.viridis

    ind = np.where(np.mean(embeddings,axis=0)>0)
    embedding_small = embeddings[:,ind].squeeze()

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(embedding_small.shape[1]*2)

    for i in range(embedding_small.shape[1]):

            im = plot_imagemap(ax[i], embedding_small[:, i].reshape(60, 60),
                                        plot_format, clim=[0,ranges[i]])

    for i in range(number):

        for j in range(len(ranges)):

            value = np.linspace(0,ranges[j], number)

            if i == 0:
                gen_value = np.zeros((embeddings.shape[1]))
            else:
                idx = find_nearest(embedding_small[:,j],value[i], average_number)
                gen_value = np.mean(embeddings[idx], axis=0)
                gen_value[j] = value[i]

            generated = model.predict(np.atleast_2d(gen_value)).squeeze()
            ax[j+embedding_small.shape[1]].plot(voltage, generated, color=cmap((i+1)/number))
            ax[j+embedding_small.shape[1]].set_ylim(y_scale)
            ax[j+embedding_small.shape[1]].set_yticklabels('')
            ax[j+embedding_small.shape[1]].set_xlabel('Voltage (V)')
            if j == 0:
                ax[j+embedding_small.shape[1]].set_ylabel('Piezoresponse (Arb. U.)')


            plt.tight_layout(pad=1)

    savefig(folder + '/generated_loops',  printing)

def plot_generator_resonance_results(
    model,
    model_piezo,
    index,
    embedding,
    ranges,
    number,
    averaging_number,
    plot_subselect,
    embedding_piezo,
    voltage,
    resonance_cleaned,
    plot_format,
    printing,
    folder,
    scales=None,
    name_prefix = ''):

    cmap = plt.cm.viridis
    # Defines the colorlist
    cmap2 = plt.get_cmap('plasma')


    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(len(index)*3, mod = len(index))
    #shifts the voltage
    shift_voltage = roll_and_append(voltage)

    for i, index_ in enumerate(index):

        # plots the image map of the region
        im = plot_imagemap(ax[i],embedding[:,index_], plot_format,
                         clim=[0,ranges[index_]])


    # loops around the number of example loops
    for i in range(number):

        # loops around the selected index
        for j, index_ in enumerate(index):

            # builds a linear space vector where we tune the embedding value
            value = np.linspace(0,ranges[index_],number)

            # plots the average loop (all embedding = 0)
            if i == 0:
                gen_value = np.zeros((len(ranges)))
            else:
                # finds a select number of indices with a value closest to the selected value
                idx = find_nearest(embedding[:,index_],value[i],averaging_number)
                # computes the mean
                gen_value = np.mean(embedding[idx],axis=0)
                # replaces the value with the selected value
                gen_value[index_] = value[i]
                # finds the embedding of the piezoelectric loop (finds points closest to the average)
                gen_value_piezo = np.mean(embedding_piezo[idx],axis=0)
            if i in plot_subselect[j]:
                # generates the curves
                generated = model.predict(np.atleast_2d(gen_value)).squeeze()
                generated_piezo = model_piezo.predict(np.atleast_2d(gen_value_piezo)).squeeze()

                # connects the first and last point
                generated = roll_and_append(generated)

                # rescales the data back to the original data space
                generated *= np.std(resonance_cleaned.reshape(-1))
                generated += np.mean(resonance_cleaned.reshape(-1))

                # plots the resonance curves
                ax[j+len(index)].plot(shift_voltage, generated,
                                      color = cmap((i+1)/number), linewidth=3)
                # plots the piezoresponse curves
                ax[j+len(index)*2].plot(voltage, generated_piezo,
                                  color = cmap((i+1)/number), linewidth=3)

                if j == 0:
                    ax[j+len(index)].set_ylabel('Resonance (kHz)')
                    ax[j+len(index)*2].set_ylabel('Piezoresponse (Arb. U.)')
                else:
                    ax[j+len(index)].set_yticklabels('')

                ax[j+len(index)*2].set_yticklabels('')

                ax[j+len(index)].set_xticklabels('')
                ax[j+len(index)*2].set_xlabel('Voltage (V)')

                if scales is None:
                    pass
                else:
                    ax[j+len(index)].set_ylim(scales[0])
                    ax[j+len(index)*2].set_ylim(scales[1])

    plt.tight_layout(pad=0)

    savefig(folder + '/' + name_prefix + '_generated_autoencoded_result',
            printing)

def roll_and_append(data):
    data = np.roll(data,data.shape[0]//4)
    data = np.append(data, data[0])

    return data

def generate_generator_images_resonance(model,
                                        index_c,index_a,
                                        embedding_c, voltage, embedding_a,
                                        ranges_c,ranges_a,
                                        number,averaging_number, resonance_decode,
                                        plot_format,
                                        printing,
                                        folder,
                                        graph_layout = [6,4]):

    # Defines the colorlist
    cmap = plt.get_cmap('viridis')

    # Loop for each of the example graphs
    for i in tqdm(range(number)):

        # builds the figure
        fig, ax = plt.subplots(graph_layout[0] // graph_layout[1]  + (graph_layout[0]  % graph_layout[1] > 0), graph_layout[1],
                                 figsize=(3 * graph_layout[1], 3 * (graph_layout[0]  // graph_layout[1] + (graph_layout[0]  % graph_layout[1] > 0))))
        ax = ax.reshape(-1)

        plot_single_generator_result_resonance(resonance_decode,
                                                ax[:graph_layout[0]//2],
                                                i,
                                                index_a,
                                                embedding_a,
                                                ranges_a,
                                                 averaging_number,
                                                number,
                                                voltage,
                                                plot_format)

        plot_single_generator_result_resonance(resonance_decode,
                                                ax[graph_layout[0]//2:],
                                                i,
                                                index_c,
                                                embedding_c,
                                                ranges_c,
                                                averaging_number,
                                                number, voltage,
                                                plot_format)

        savefig(pjoin(folder,f'{i:04d}_maps'), printing)
        plt.close(fig)

def plot_single_generator_result_resonance(model, ax, i, index,
            embedding, ranges,averaging_number, number, voltage,
                                           plot_format):

    for j, index_ in enumerate(index):

        im = plot_imagemap(ax[j],embedding[:,index_], plot_format,
                         clim=[0,ranges[index_]])

        value = np.linspace(0,ranges[index_],number)

        mean_loop = model.predict(np.atleast_2d(np.zeros((len(ranges))))).squeeze()

        if i == 0:
            gen_value = np.zeros((len(ranges)))
        else:
            idx = find_nearest(embedding[:,index_],value[i],averaging_number)
            gen_value = np.mean(embedding[idx],axis=0)
            gen_value[index_] = value[i]

        generated = model.predict(np.atleast_2d(gen_value)).squeeze()
        ax[j+len(index)].plot(voltage, generated, color = cmap((i+1)/number))
        ax[j+len(index)].set_ylim(-2,2)
        ax[j+len(index)].set_yticklabels('')
        ax[j+len(index)].plot(voltage, mean_loop, color = cmap((0+1)/number))

        pos = ax[j+len(index)].get_position()

def impose_topography(topodata, data):

    empty_mat = np.empty((int(20*5)+20,128*4))
    empty_mat[:,:] = np.nan

    # loops around the x axis
    for j in range(topodata.shape[1]):

        # resets the count for each cloumn
        count = 0

        for i in range(topodata.shape[0]):

            if np.abs(topodata[i,j])>.5:
                empty_mat[count:count+5,j*4:j*4+4]=data[i,j]
                count+=5
            else:
                empty_mat[count:count+4,j*4:j*4+4]=data[i,j]
                count+=4

    return empty_mat


def plot_phase_field_switching(Phase_field_information, printing):

    ax = [None]*6*3

    for tip in Phase_field_information['tips']:

        # makes the figures
        fig = plt.figure(figsize=(6, 2*6))

        # builds the axes for the figure
        for j in range(6):
            for i in range(3):
                ax[i+(j*3)] = plt.subplot2grid((6, 6), (j, i*2),
                                             colspan=2, rowspan=1)

        timedata = np.genfromtxt(Phase_field_information['folder']['time_series'] + tip + '/timedata.dat')
        files = np.sort(glob.glob(Phase_field_information['folder']['polarization'] +'data/' + tip +'/*.dat'))
        files_energy = np.sort(glob.glob(Phase_field_information['folder']['energy'] + tip +'/*.dat'))

        for i, index in enumerate(Phase_field_information['time_step']):

            # gets the tip position
            pos = Phase_field_information['tip_positions'][tip]['pos']

            data = np.genfromtxt(files[index], skip_header=1, skip_footer=3)
            data_energy = np.genfromtxt(files_energy[index], skip_header=1, skip_footer=3)

            idz = np.where(data[:,2] == Phase_field_information['graph_info']['top'])
            idy = np.where(data[:,1] == Phase_field_information['graph_info']['y_cut'])
            ylim = Phase_field_information['graph_info']['y_lim']
            xlim = Phase_field_information['graph_info']['x_lim']

            topodata = np.rollaxis(data[idy,2+3].reshape(128,-1),1)

            for k, labels in enumerate(Phase_field_information['labels']):

                if labels == 'Polarization Z':
                    data_pz = np.rollaxis(data[idy,5].reshape(128,-1),1)
                    data_out = impose_topography(topodata, data_pz)
                else:
                    data_energy_ = np.rollaxis(data_energy[idy,2+k].reshape(128,-1),1)
                    data_out = impose_topography(topodata, data_energy_)

                im = ax[i+(3*k)].imshow(data_out,clim=Phase_field_information['graph_info']['clim'][labels],
                                      cmap=mymap)


                ax[i+(3*k)].set_ylim(ylim[0],ylim[1])
                ax[i+(3*k)].set_xlim(xlim[0],xlim[1])

                #defines the tip height based on the highest value
                tip_height = np.max(np.argwhere(np.isfinite(data_out[:,pos[0]*4])))


                ax[i+(3*k)].annotate('',xy=(pos[0]*4, tip_height), xycoords='data',
                     xytext=(pos[0]*4,tip_height+20), textcoords='data',
                     arrowprops=dict(arrowstyle="wedge", facecolor = 'k'))

                if i == 1:
                    ax[i+(3*k)].set_title(labels)
                elif i == 2:
                    divider = make_axes_locatable(ax[i+(3*k)])
                    cax = divider.append_axes('right', size='10%', pad=0.05)
                    cbar = plt.colorbar(im, cax=cax)
                    tick_locator = ticker.MaxNLocator(nbins=3)
                    cbar.locator = tick_locator
                    cbar.update_ticks()
                    if k != 0:
                        cbar.set_label('$J/m^{3}$', rotation=270, labelpad=6)

        for axes in ax:
            #Removes axis labels
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axison = False

        savefig(Phase_field_information['output_folder'] + '/' + tip + '_x_{}_y_{}'.format(pos[0],pos[1]),
                   printing)


def export_images_phase_field_movie(Phase_field_information, printing):

    for tip in Phase_field_information['tips']:

        fig, ax = layout_graphs_of_arb_number(8,4)

        timedata = np.genfromtxt(Phase_field_information['folder']['time_series'] + tip + '/timedata.dat')
        files = np.sort(glob.glob(Phase_field_information['folder']['polarization'] +'data/' + tip +'/*.dat'))
        files_energy = np.sort(glob.glob(Phase_field_information['folder']['energy'] + tip +'/*.dat'))



        phase_field_voltage = timedata[::5,2]
        fit_results = loop_fitting_function(phase_field_voltage,-1,1,0,2.5,2.5,2.5,2.5,2.5,2.5,0,0,-5,5)
        example_loop = np.concatenate((fit_results['Branch2'][0:20],
                                       fit_results['Branch1'][20:60],
                                       fit_results['Branch2'][60::]))

        current_folder = make_folder(Phase_field_information['output_folder'] + '/movie/' + tip)

        for index in range(files.shape[0]):

            fig, ax = layout_graphs_of_arb_number(8,4)

            # gets the tip position
            pos = Phase_field_information['tip_positions'][tip]['pos']

            data = np.genfromtxt(files[index], skip_header=1, skip_footer=3)
            data_energy = np.genfromtxt(files_energy[index], skip_header=1, skip_footer=3)

            idz = np.where(data[:,2] == Phase_field_information['graph_info']['top'])
            idy = np.where(data[:,1] == Phase_field_information['graph_info']['y_cut'])
            ylim = Phase_field_information['graph_info']['y_lim']
            xlim = Phase_field_information['graph_info']['x_lim']

            topodata = np.rollaxis(data[idy,2+3].reshape(128,-1),1)

            for k, labels in enumerate(Phase_field_information['labels']):

                    if k > 2:
                        inds = k+2
                    else:
                        inds = k+1

                    if labels == 'Polarization Z':
                        data_pz = np.rollaxis(data[idy,5].reshape(128,-1),1)
                        data_out = impose_topography(topodata, data_pz)
                    else:
                        data_energy_ = np.rollaxis(data_energy[idy,2+k].reshape(128,-1),1)
                        data_out = impose_topography(topodata, data_energy_)

                    im = ax[inds].imshow(data_out,clim=Phase_field_information['graph_info']['clim'][labels],
                                          cmap=mymap)

                    ax[inds].set_ylim(ylim[0],ylim[1])
                    ax[inds].set_xlim(xlim[0],xlim[1])

                    #defines the tip height based on the highest value
                    tip_height = np.max(np.argwhere(np.isfinite(data_out[:,pos[0]*4])))


                    ax[inds].annotate('',xy=(pos[0]*4, tip_height), xycoords='data',
                         xytext=(pos[0]*4,tip_height+20), textcoords='data',
                         arrowprops=dict(arrowstyle="wedge", facecolor = 'k'))

                    ax[inds].set_title(labels)

            ax[0].plot(phase_field_voltage, example_loop)
            ax[0].plot(phase_field_voltage[index], example_loop[index],'o')
            ax[0].set_xlabel('Voltage')
            ax[0].set_ylabel('Polarization')
            ax[0].set_yticks([])

            ax[4].plot(phase_field_voltage)
            ax[4].plot(index,phase_field_voltage[index],'o')
            ax[4].set_xlabel('Time Step')
            ax[4].set_ylabel('Voltage (V)')


            for j, axes in enumerate(ax):

                if j not in [0,4]:
                    #Removes axis labels
                    axes.set_xticks([])
                    axes.set_yticks([])
                    axes.axison = False



            savefig(current_folder + '/Image_{0:03d}'.format(index+1),
                       printing)

            plt.close(fig)

def loop_fitting_function(V,a1,a2,a3,b1,b2,b3,b4,b5,b6,b7,b8,Au,Al):

    S1 = ((b1+b2)/2) + ((b2-b1)/2)*erf((V-b7)/b5)
    S2 = ((b4+b3)/2) + ((b3-b4)/2)*erf((V-b8)/b6)
    Branch1 = (a1+a2)/2 + ((a2-a1)/2)*erf((V-Au)/S1)+a3*V
    Branch2 = (a1+a2)/2 + ((a2-a1)/2)*erf((V-Al)/S2)+a3*V


    return {'Branch1':Branch1 , 'Branch2':Branch2}

def plot_line_with_color(ax, y, z):

    x = np.linspace(0,y.shape[0],y.shape[0])

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    #lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc = LineCollection(segments, cmap=plt.get_cmap('viridis'))
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-4e-9,4e-9)
#
#def plot_embedding_and_line_trace(ax, map_number, topo_map,
#                                  embedding_map,
#                                  climit,
#                                 resize_shape = 900,
#                                 line_value = 550,
#                                 number=3):
#
#    image, scalefactor  = rotate_and_crop(embedding_map.reshape(60, 60),
#                                                angle=60.46, frac_rm = 0.17765042979942694)
#
#    im = plot_imagemap(ax[map_number + number], image,
#                         clim=climit,
#                         add_scalebar=[2000-(2000*248/1024),500])
#
#    resize_embedding = set_value_scale(image.reshape(-1),climit)
#    image_resize =  imresize(resize_embedding,(topo_map.shape[0],topo_map.shape[1]))
#
#    plot_line_with_color(ax[map_number], np.mean(topo_map,axis=0),
#                     np.mean(resize_embedding.reshape(np.int(np.sqrt(resize_embedding)),np.int(np.sqrt(resize_embedding))),axis=0))
#
def set_value_scale(data, clim):
    data[np.where(data<clim[0])] = clim[0]
    data[np.where(data>clim[1])] = clim[1]
    return data


def plot_PFM_and_line_trace(signals, imported, printing, folder,name='/PFM_image'):
    # makes the figures
    fig = plt.figure(figsize=(9, 6))

    # builds the axes
    ax1 = plt.subplot2grid((4, 6), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((4, 6), (2, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid((4, 6), (3, 0), colspan=2, rowspan=1)
    ax4 = plt.subplot2grid((4, 6), (0, 2), colspan=2, rowspan=2)
    ax5 = plt.subplot2grid((4, 6), (0, 4), colspan=2, rowspan=2)
    ax6 = plt.subplot2grid((4, 6), (2, 2), colspan=2, rowspan=2)
    ax7 = plt.subplot2grid((4, 6), (2, 4), colspan=2, rowspan=2)

    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7)



    # Plots each of the graphs
    for i, (signal, values) in enumerate(signals.items()):

        if 'Trace' not in signal:

            # Plots the graph
            im = axes[i].imshow(
                np.flipud(imported['data'][values['data_loc']].T), cmap='plasma')

            # Sets the scales
            if values['c_lim']:
                im.set_clim(values['c_lim'])

            # labels figures
            labelfigs(axes[i], i)

            # Removes axis labels
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])

            # Sets ticks
            axes[i].set_xticks(np.linspace(
                0, imported['data'][values['data_loc']].shape[0], 5))
            axes[i].set_yticks(np.linspace(
                0, imported['data'][values['data_loc']].shape[1], 5))

            # adds the scalebars to figure
            add_scalebar_to_figure(axes[i], 2000, 500)

        else:

            # collects the data for the line trace
            line_trace = imported['data'][values['data_loc']]

            # plots the line trace
            plt1 = axes[i].plot(line_trace[:, 0] * 1e6 -
                                values['shift'], line_trace[:, 1] * 1e9, 'k')

            if 'Small' in signal:
                axes[i].set_xlabel('Distance (${\mu}$m)')

            # sets the axes for the line trace plot
            axes[i].set_ylabel('Height (nm)')
            axes[i].set_xlim(values['x_lim'])
            axes[i].set_ylim(values['y_lim'])
            axes[i].set_xticks(np.linspace(0, values['x_lim'][1], 3))
            axes[i].set_yticks(np.arange(values['y_lim'][0],
                                         values['y_lim'][1] + .1, 1))
            axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            # labels figures
            labelfigs(axes[i], i, style='b')

    plt.tight_layout()

    savefig(folder + name, printing)

def plot_RSM(data, printing, folder, name='/002_RSM'):
    # Creates the figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    # Plots the graph
    Plt1 = ax1.imshow(np.flipud(np.log(data['data']['vq'])), cmap='viridis')

    # Formats the graph
    Plt1.set_clim([3, 8])
    ax1.set_xticks(np.linspace(0, 1000, 9))
    ax1.set_xticklabels(np.linspace(-4, 4, 9))
    ax1.set_yticks(np.linspace(777.77777 / 5, 777.7777777777778, 4))
    ax1.set_ylim([1000, 200])
    ax1.set_yticklabels(np.linspace(.51, .48, 4))
    ax1.set_xlabel('Q${_x}$ (1/${\AA}$)')
    ax1.set_ylabel('Q${_y}$ (1/${\AA}$)')
    ax1.set_facecolor([0.26700401,  0.00487433,  0.32941519])

    # Saves the figure
    savefig(folder + name,printing)

def plot_PFM_images(signals, imported, printing, folder, title):
    # Defines the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Adds title to the figure
    fig.suptitle(title, fontsize=16,
                 y=1, horizontalalignment='center')

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signals.items()):

        image, scale = rotate_and_crop(np.flipud(imported['data'][values['data_loc']].T))
        # Plots the graph
        im = axes[i].imshow(image, cmap='plasma')

        # Sets the scales
        im.set_clim(values['c_lim'])

        # Adds titles to the graphs
        axes[i].set_title(signal, fontsize=14)

        # labels figures
        labelfigs(axes[i], i)

        # Removes axis labels
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])

        # Sets ticks
        axes[i].set_xticks(np.linspace(0, 1024/scale, 5))
        axes[i].set_yticks(np.linspace(0, 1024/scale, 5))

        add_scalebar_to_figure(axes[i], 2000/scale, 500)

    savefig(folder + '/' + title,
                    printing)

def update_decoder(model, weights):
    # builds the resonance decoding model
    decode = Sequential()
    decode.add(BatchNormalization(input_shape=(16,)))
    decode.add(RepeatVector(96))
    decode.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(96, 1)))
    decode.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(96, 1)))
    decode.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(96, 1)))
    decode.add(Bidirectional(LSTM(64, return_sequences=True),input_shape=(96, 1)))
    decode.add(TimeDistributed(Dense(1, activation='linear')))

    decode.compile(Adam(3e-5), loss='mse')

    decode.load_weights(weights)

    # Sets the layers to match the training model
    model.layers[10].set_weights((decode.layers[0].get_weights()))
    model.layers[11].set_weights((decode.layers[1].get_weights()))
    model.layers[12].set_weights((decode.layers[2].get_weights()))
    model.layers[14].set_weights((decode.layers[3].get_weights()))
    model.layers[16].set_weights((decode.layers[4].get_weights()))
    model.layers[18].set_weights((decode.layers[5].get_weights()))
    model.layers[20].set_weights((decode.layers[6].get_weights()))

    return model, decode

def plot_loss_and_reconstruction(model_folder,data,
                                                    model,signal,signal_info,
                                                    printing,folder, i=None):

    voltage = data['raw']['voltage']
    if i is None:
        i = np.random.randint(0,data['normalized'][signal].shape[0])

    # creates the figures and axes in a pretty way
    fig, ax = layout_graphs_of_arb_number(3)

    loss = np.rollaxis(np.loadtxt(model_folder + '/log.csv',
                                            skiprows=1, delimiter=','),1)

    ax[0].plot(loss[0],
               loss[1],'r',
               loss[2],'k')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')

    ax[1].plot(voltage,data['normalized'][signal][i],'k')
    ax[1].plot(voltage, model.predict(np.atleast_3d(data['normalized'][signal][i])).squeeze(),'r')

    # sets the axis titles
    ax[1].set_xlabel('Voltage')
    ax[1].set_ylabel(signal_info[signal]['units'])
    ax[1].set_yticks(signal_info[signal]['y_tick'])
    ax[1].set_ylim([-2,2])

    # resizes the figure
    set_axis_aspect(ax[1])

    ax[2].plot(voltage,data['normalized']['val_' + signal][i],'k')
    ax[2].plot(voltage,
               model.predict(np.atleast_3d(data['normalized']['val_' + signal][i])).squeeze(),'r')

    # sets the axis titles
    ax[2].set_xlabel('Voltage')
    ax[2].set_yticks(signal_info[signal]['y_tick'])
    ax[2].set_ylim([-2,2])

    # resizes the figure
    set_axis_aspect(ax[2])

    plt.tight_layout(pad=0)

    savefig(folder + '/' + signal + '_{}'.format(i), printing)


def train_model(run_id, model, data, data_val, folder,
                batch_size=1800, epochs=25000, seed=42):

    time = datetime.datetime.now()

    np.random.seed(seed)

    run_id = make_folder(folder + '/{0}_{1}_{2}_{3}h_{4}m'.format(time.month,
                                                                      time.day, time.year,
                                                                      time.hour, time.minute) + '_' + run_id)

    model_name = run_id + 'start'
    keras.models.save_model(model, run_id + '/start_seed_{0:03d}.h5'.format(seed))


    filepath = run_id + '/weights.{epoch:06d}-{val_loss:.4f}.hdf5'

    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                 verbose=0, save_best_only=True,
                                                 save_weights_only=True, mode='min', period=1)

    logger = keras.callbacks.CSVLogger(run_id + '/log.csv', separator=',', append=True)

    history = model.fit(np.atleast_3d(data), np.atleast_3d(data), epochs=25000,
              batch_size=1800, validation_data=(np.atleast_3d(data_val), np.atleast_3d(data_val)),
              callbacks=[checkpoint, logger])

def export_training_images_for_movie(model, data,  model_folder, printing, plot_format,
                                                                folder,data_type='piezoresponse'):

    printing_ = printing.copy()
    plot_format_ = plot_format.copy()

    plot_format_['color_bars'] = False
    printing_['EPS'] = False

    # simple function to help extract the filename
    def name_extraction(filename):
        filename = file_list[0].split('/')[-1][:-5]
        return filename

    embedding_exported = {}

    # searches the folder and finds the files
    file_list = glob.glob(model_folder + '/weights*')
    file_list = natsorted(file_list, key=lambda y: y.lower())

    for i, file_list in enumerate(file_list):
        # loads the weights into the model
        model.load_weights(file_list)

        #Computes the low dimensional layer
        embedding_exported[name_extraction(file_list)] = get_activations(model,
                            data['normalized'][data_type],9)

        # plots the embedding maps
        _ = plot_embedding_maps(embedding_exported[name_extraction(file_list)], printing_, plot_format_,
                               folder,  filename='./epoch_{0:04}'.format(i))

        # Closes the figure
        plt.close(_)


def plot_embedding_and_line_trace(ax, map_number, topo_map,
                                  embedding_map,
                                  climit,
                                  plot_format,
                                 resize_shape = 900, number=3):

    rotated_embedding, scale_factor = rotate_and_crop(
        embedding_map.reshape(60,60),
                        angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
    resize_embedding = resize(rotated_embedding, (resize_shape, resize_shape))
    hold = np.copy(plot_format['rotation'] )
    plot_format['rotation'] = False
    plot_imagemap(ax[map_number + number],resize_embedding,plot_format,
                         clim=climit)

    resize_embedding = set_value_scale(resize_embedding.reshape(-1),
                                       climit)

    plot_line_with_color(ax[map_number], np.mean(topo_map,axis=0),
                     np.mean(resize_embedding.reshape(900,900),axis=0), scale_factor)
    plot_format['rotation'] = hold

def plot_line_with_color(ax, y, z, scale_factor):

    x = np.linspace(0,y.shape[0],y.shape[0])

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    #lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc = LineCollection(segments, cmap=plt.get_cmap('viridis'))
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-4e-9,4e-9)
    ax.set_xticklabels(np.linspace(0,500*((2000*scale_factor)//500),5).astype(int))
    ax.set_xlabel('Position (nm)')
    ax.set_ylabel('Height (nm)')
