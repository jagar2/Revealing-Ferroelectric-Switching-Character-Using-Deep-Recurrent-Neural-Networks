"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

from numpy.core.multiarray import ndarray
from .format import *
import numpy as np
from ..util.file import *
from ..util.core import *
from ..analysis.machine_learning import *
from ..analysis.rnn import *
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from natsort import natsorted, ns
from skimage.transform import resize
from matplotlib.collections import LineCollection
from tqdm import tqdm
cmap = plt.get_cmap('viridis')


# Defines a set of custom color maps
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


def cleaned_data(data, i, printing, folder='./'):
    """
    Plots the cleaned data

    Parameters
    ----------
    data : numpy, array
        data to plot
    i : numpy, int
        pixel to plot
    printing : dict
        information for printing the figures
    folder : string (optional)
        folder to save the image

    """

    # layout the figure
    fig, ax = layout_fig(5, 5)
    # extracts the voltage
    voltage = data['raw']['voltage']
    count = 0

    # build a looping function to display all the cleaned data
    # Plots each of the graphs
    for j, (signal, values) in enumerate(data['raw'].items()):

        if j in [1, 2, 3, 4, 5]:
            # plots the various signals
            ax[count].plot(voltage, data['interpolated'][signal][i], 'k')
            ax[count].plot(voltage, data['sg_filtered'][signal][i], 'r')

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


def pfm_w_line_trace(signals, imported, printing, folder, name='/PFM_image', colorbar_shown = False):
    """
    Plots PFM image with line trace

    Parameters
    ----------
    signals : dict
        information for the graph
    imported : dict
        data to plot
    printing : dict
        information for printing the figures
    folder : string
        folder to save the image
    name : string
        filename to save the image
    colorbar_shown : bool
        shows the colorbar


    """

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
            scalebar(axes[i], 2000, 500)

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

    if colorbar_shown:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax,ticks = None)

    savefig(folder + name, printing)


def rsm(data, printing, folder, plot_format, name='/002_RSM'):
    """
    Plots RSM image

    Parameters
    ----------
    data : dict
        raw RSM data
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        folder to save the image
    name : string
        filename to save the image
    plot_format : dict
        list of settings for the plot format

    """

    # Creates the figure and axes
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)

    # Plots the graph
    im = ax1.imshow(np.flipud(np.log(data['data']['vq'])), cmap='viridis')

    # Formats the graph
    im.set_clim([3, 8])
    ax1.set_xticks(np.linspace(0, 1000, 9))
    ax1.set_xticklabels(np.linspace(-4, 4, 9))
    ax1.set_yticks(np.linspace(777.77777 / 5, 777.7777777777778, 4))
    ax1.set_ylim([1000, 200])
    ax1.set_yticklabels(np.linspace(.51, .48, 4))
    ax1.set_xlabel('Q${_x}$ (1/${\AA}$)')
    ax1.set_ylabel('Q${_y}$ (1/${\AA}$)')
    ax1.set_facecolor([0.26700401,  0.00487433,  0.32941519])

    if plot_format['color_bars']:
        colorbar(ax1,im, label = 'Counts', num_format='%0.0f')

    # Saves the figure
    savefig(folder + name, printing)


def pfm(signals, imported, printing, folder, title):
    """
    Plots RSM image

    Parameters
    ----------
    signals : dict
        information about the plots
    imported : dict
        data to plot
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        folder to save the image
    title : string
        title of the figure and prefix for the filename

    """

    # Defines the figure and subplots
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Adds title to the figure
    fig.suptitle(title, fontsize=16,
                 y=1, horizontalalignment='center')

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signals.items()):

        # Plots the graph
        im = axes[i].imshow(
            np.flipud(imported['data'][values['data_loc']].T), cmap='plasma')

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
        axes[i].set_xticks(np.linspace(0, 1024, 5))
        axes[i].set_yticks(np.linspace(0, 1024, 5))

        scalebar(axes[i], 2000, 500)

    savefig(folder + '/' + title,
            printing)


def band_excitation(data, signal_clim,  plot_format,
                    printing, folder_=''):
    """
    Function which prints all of the BE images

    Parameters
    ----------
    data : numpy, float
        raw data to plot
    signal_clim  : dictionary
        Instructions for extracting the data and plotting the data
    plot_format  : dictionary
        sets the format for the plots
    printing :dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder_ : string
        path to save the files
    """

    # Graphs and prints all figures
    for (signal_name, signal), colorscale in signal_clim.items():

        # makes the folder to save the files
        folder = make_folder(folder_ + '/{}'.format(signal_name))

        # Cycles around each loop
        for cycle in (1, 2):

            # Builds data name
            field = 'Out{0}{1}_mixed'.format(signal, cycle)

            # Displays loop status
            print('{0} {1}'.format(signal_name, cycle))

            # Loops around each voltage step
            for i in range(data[field].shape[2]):

                # Defines the figure and axes
                fig, ax1 = plt.subplots(figsize=(3, 3))

                # crops and rotates the image
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
                    scalebar(ax1, plot_format['scalebar'][0] * scale_factor,
                             plot_format['scalebar'][1])

                # Generates the filename
                filename = '{0}{1}_{2:03d}'.format(signal, cycle, i)

                # Saves the figure
                savefig(folder + '/' + filename,
                        printing)

                # Closes the figure
                plt.close(fig)


def band_excitation_movie(data, signals_clim,
                          plot_format, printing,
                          folder='Movie Images',):
    """
    Plots raw BE data

    Parameters
    ----------
    data : dictionary
        imported collection of data
    signals_clim : list
        description of what to plot
    plot_format : dictionary
        sets the format for what to plot
    printing :dictionary
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
    voltage = np.copy(data['data']['Voltagedata_mixed'])  # type: ndarray
    voltage_steps = voltage.shape[0]
    voltage = roll_and_append(voltage)[::-1]
    voltage = np.append(voltage, voltage)

    # Cycles around each loop
    for cycle in (1, 2):

            # Loops around each voltage step
        for i in range(voltage_steps):

                # Defines the axes positions
            fig = plt.figure(figsize=(8, 12))
            ax1 = plt.subplot2grid((3, 2), (0, 0))
            ax2 = plt.subplot2grid((3, 2), (0, 1))
            ax3 = plt.subplot2grid((3, 2), (1, 0))
            ax4 = plt.subplot2grid((3, 2), (1, 1))
            ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
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
                image, scale_frac = rotate_and_crop(data['data'][field][:, :, i], plot_format['angle'],
                                                    plot_format['frac_rm'])
                im = axes[j].imshow(image)
                axes[j].set_title(signal_name)
                im.set_clim(colorscale)

                # Sets the color bars
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

            savefig(folder + '/' + filename,
                    printing)

            # Closes the figure
            plt.close(fig)


def band_excitation_spectra(x, y, cycle, data, signals, printing, folder, cmaps='inferno'):
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
    data : dictionary
        raw Band Excitation Data to plot
    signals : list
        description of what to plot
    printing : dictionary
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
        axes[i].set_xticks(np.linspace(-1 * (max(data['Voltagedata_mixed']) // 5 * 5),
                                       max(data['Voltagedata_mixed']) // 5 * 5, 7))
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

            # Computes and shapes the matrix for the piezoresponse
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
                printing)


def loopfits(data, signal_clim, printing, folder, plot_format):
    """
    Saves figure

    Parameters
    ----------
    data : dictionary
        dictionary containing the loop fitting results
    signal_clim : list
        description of what to plot
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string
        folder where the images will be saved
    plot_format : dictionary
        sets the format for what to plot
    """

    # Defines the figure and axes
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    axes = axes.reshape(30)

    # Plots each of the graphs
    for i, (signal, values) in enumerate(signal_clim.items()):

        # Sets the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0, 59, 5))
        axes[i].set_yticks(np.linspace(0, 59, 5))
        axes[i].set_title('{0}'.format(values['label']))
        axes[i].set_facecolor((.55, .55, .55))

        field = '{}'.format(values['data_loc'])

        if plot_format['rotation']:
            image, scalefactor = rotate_and_crop(
                data[field], plot_format['angle'], plot_format['frac_rm'])
        else:
            scalefactor = 1

        # Plots the graphs either abs of values or normal
        if i in {13, 20, 21, 22, 23}:

            im = imagemap(axes[i], np.abs(data[field]),
                          plot_format, clim=values['c_lim'])

        else:

            im = imagemap(axes[i], data[field],
                          plot_format, clim=values['c_lim'])

        # labels figures
        labelfigs(axes[i], i)

    # Deletes unused figures
    fig.delaxes(axes[28])
    fig.delaxes(axes[29])

    # Saves Figure
    plt.tight_layout(pad=0, h_pad=-20)
    savefig(folder + '/loopfitting_results', printing)


def imagemap(ax, data, plot_format, clim=None):
    """
    Plots an image map

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
        data = data.reshape(np.sqrt(data.shape[0]).astype(
            int), np.sqrt(data.shape[0]).astype(int))

    if plot_format['color_map'] is None:
        cmap = plt.get_cmap('viridis')
    else:
        cmap = plt.get_cmap(plot_format['color_map'])

    if plot_format['rotation']:
        data, scalefactor = rotate_and_crop(data, angle=plot_format['angle'],
                                            frac_rm=plot_format['frac_rm'])
    else:
        scalefactor = 1

    if clim is None:
        im = ax.imshow(data,  cmap=cmap)
    else:
        im = ax.imshow(data, clim=clim, cmap=cmap)

    ax.set_yticklabels('')
    ax.set_xticklabels('')

    if plot_format['color_bars']:
        # adds the colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%.1e')

    # adds scalebar to figure
    if plot_format['add_scalebar'] is not False:
        scalebar(ax, plot_format['scalebar'][0] *
                 scalefactor, plot_format['scalebar'][1])

    return im


def pca_results(pca, data, signal_info, printing, folder, plot_format, signal,
                verbose=False,
                letter_labels=True, filename='./PCA_maps',
                num_of_plots=True):
    """
    Plots the pca results

    Parameters
    ----------
    pca : object
        results from the pca
    data  : numpy, float
        data
    signal_info  : dict
        controls the design of the plots
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string
        folder where the images will be saved
    plot_format : dictionary
        sets the format for what to plot
    signal : string
        name of the signal to plot
    verbose : bool, optional
        selects to show output text
    filename : string, optional
        filename for the output file
    num_of_plots = int, optional
        sets the number of pc to show (if true shows all computed)
    """

    voltage = data['raw']['voltage']
    loops = data['sg_filtered'][signal]
    min_ = np.min(pca.components_.reshape(-1))
    max_ = np.max(pca.components_.reshape(-1))
    count = 0

    if num_of_plots == True:
        num_of_plots = pca.n_components_

    # stores the number of plots in a row
    mod = num_of_plots // (np.sqrt(num_of_plots) // 1).astype(int)
    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(num_of_plots * 2, mod=mod)

    # resizes the array for hyperspectral data
    if loops.ndim == 3:
        original_size = loops.shape[0]
        loops = loops.reshape(-1, loops.shape[2])
        verbose_print(verbose, 'shape of data resized to [{0} x {1}]'.format(
            loops.shape[0], loops.shape[1]))
    elif loops.ndim == 2:
        original_size = np.sqrt(loops.shape[0]).astype(int)
    else:
        raise ValueError("data is of an incorrect size")

    # computes the PCA maps
    PCA_maps = weights_as_embeddings(
        pca, loops, num_of_components=num_of_plots)

    # Formats figures
    for i, ax in enumerate(ax):

        # Checks if axes is an image or a plot
        if i // mod % 2 == 0:
            pc_number = i - mod * (i // (mod * 2))
            im = imagemap(ax, PCA_maps[:, pc_number], plot_format)
            # labels figures
            labelfigs(ax, i, string_add='PC {0:d}'.format(
                pc_number + 1), loc='bm')
            # if plot_format['add_scalebar']:
            #    add_scalebar_to_figure(ax, plot_format['scalebar'][0], plot_format['scalebar'][1])

        else:

            # Plots the PCA egienvector and formats the axes
            ax.plot(voltage, pca.components_[
                    i - mod - ((i // mod) // 2) * mod], 'k')

            # Formats and labels the axes
            ax.set_xlabel('Voltage (V)')
            ax.set_ylabel(signal_info[signal]['units'])
            ax.set_yticklabels('')
            ax.set_ylim([min_, max_])

            if signal_info[signal]['pca_range'] is not None:
                ax.set_ylim(signal_info[signal]['pca_range'])

        # labels figures
        if letter_labels:
            if i // mod % 2 == 0:
                labelfigs(ax, count)
                count += 1

        set_axis_aspect(ax)

    plt.tight_layout(pad=0, h_pad=0)

    savefig(folder + '/' + filename, printing)


def NMF(voltage, nmf,
        printing, plot_format,
        signal_info,
        folder='./',
        letter_labels=False,
        custom_order=None):
    """
    Plots the nmf results

    Parameters
    ----------
    voltage : numpy, array
        voltage array
    nmf : object
        results from the nmf
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    plot_format  : dict
        format for the plots
    signal_info  : dict
        controls the design of the plots
    folder  : string (optional)
        folder where the images will be saved
    letter_labels : bool, optional
        allows for user specified labels
    custom_order : array, optional
        allows the user to set a custom order for the plots
    """

    W = nmf[0]
    H = nmf[1]

    # extracts the number of maps
    num_of_plots = H.shape[1]

    image_size = np.sqrt(H.shape[0]).astype(int)

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(num_of_plots * 2, mod=num_of_plots)

    min_ = np.min(W[:, :].reshape(-1))
    max_ = np.max(W[:, :].reshape(-1))

    if custom_order is not None:
        order = custom_order

    for i, ax in enumerate(ax):

        # Checks if axes is an image or a plot
        if i // num_of_plots % 2 == 0:
            # converts axis number to index number
            k = i - ((i // num_of_plots) // 2) * num_of_plots

            im = imagemap(ax, H[:, order[i]].reshape(
                image_size, image_size), plot_format)

            # labels figures
            if letter_labels:
                if i // num_of_plots % 2 == 0:
                    labelfigs(ax, k)
        else:
            # converts axis number to index number
            k = i - num_of_plots - ((i // num_of_plots) // 2) * num_of_plots

            ax.plot(voltage, W[:, order[k]], 'k')

            ax.set_xlabel('Voltage (V)')
            ax.set_ylim([min_, max_])
            ax.set_ylabel(signal_info['units'])
            ax.set_yticklabels('')

            set_axis_aspect(ax)

    plt.tight_layout(pad=0, h_pad=-10)

    savefig(folder + '/nmf_' + signal_info['symbol'], printing)


def hierarchical_clustering(cluster_results, names, plot_format):
    """
    Plots the nmf results

    Parameters
    ----------
    cluster_results : numpy, array
        results from clustering
    names : dict
        labels for figures to cluster
    plot_format  : dict
        format for the plots

    """

    combined_map, cluster_ca, c_map, a_map = cluster_results

    # Defines the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))

    # Loops around all the clusters found
    for i, name in enumerate(names):

        (title, cluster_map) = name

        # sets the order of the plots
        if cluster_map == 'cluster_ca':
            i = 0
        elif cluster_map == 'a_map':
            i = 2
        elif cluster_map == 'c_map':
            i = 1

        size_image = np.sqrt(c_map.shape[0]).astype(int) - 1

        num_colors = len(np.unique(eval(cluster_map)))
        scales = [np.max(eval(cluster_map)) - (num_colors - .5),
                  np.max(eval(cluster_map)) + .5]

        # Formats the axes
        axes[i].set_xticklabels([])
        axes[i].set_yticklabels([])
        axes[i].set_xticks(np.linspace(0, size_image, 5))
        axes[i].set_yticks(np.linspace(0, size_image, 5))
        axes[i].set_title(title)
        axes[i].set_facecolor((.55, .55, .55))

        if plot_format['rotation']:
            image, scalefactor = rotate_and_crop(eval(cluster_map).reshape(size_image + 1, size_image + 1),
                                                 angle=plot_format['angle'],
                                                 frac_rm=plot_format['frac_rm'])
        else:
            scalefactor = 1
            image = eval(cluster_map).reshape(size_image + 1, size_image + 1)

        # Plots the axes
        im = axes[i].imshow(image,
                            cmap=eval(f'cmap_{num_colors}'), clim=scales)

        labelfigs(axes[i], i, loc='br')
        scalebar(axes[i], plot_format['scalebar'][0] * scalefactor,
                 plot_format['scalebar'][1], loc='br')

        # Formats the colorbar
        divider = make_axes_locatable(axes[i])
        cax = divider.append_axes('right', size='10%', pad=0.05)
        cbar = plt.colorbar(im, cax=cax, format='%d')
        cbar.set_ticks([])


def clustered_hysteresis(voltage,
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
    hysteresis  : numpy array, float
        hysteresis loops to plot
    cluster_results :  numpy array, int
        array containing the results from clustering
    plot_format  : dict
        sets the plot format for the images
    signal_info  : dict
        controls the design of the plots
    channel  : string
        sets the channel to plot
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string (optional)
        folder where the images will be saved

    """

    combined_map, cluster_ca, c_map, a_map = cluster_results

    # organization of the raw data

    num_pix = np.sqrt(combined_map.shape[0]).astype(int)
    num_clusters = len(np.unique(combined_map))

    # Preallocates some matrix
    clustered_maps = np.zeros((num_clusters, num_pix, num_pix))
    clustered_ave_hysteresis = np.zeros((num_clusters, hysteresis.shape[1]))

    cmap_ = eval(f'cmap_{num_clusters}')

    # Loops around the clusters found
    for i in range(num_clusters):

        # Stores the binary maps
        binary = (combined_map == i + 1)
        clustered_maps[i, :, :] = binary.reshape(num_pix, num_pix)

        # Stores the average piezoelectric loops
        clustered_ave_hysteresis[i] = np.mean(
            hysteresis[binary], axis=0)

    fig, ax = layout_fig(num_clusters + 1)

    for i in range(num_clusters + 1):

        if i == 0:

            scales = [np.max(combined_map) - (num_clusters - .5),
                      np.max(combined_map) + .5]

            # Formats the axes
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])
            ax[i].set_xticks(np.linspace(0, num_pix, 5))
            ax[i].set_yticks(np.linspace(0, num_pix, 5))
            ax[i].set_facecolor((.55, .55, .55))

            if plot_format['rotation']:
                image, scalefactor = rotate_and_crop(combined_map.reshape(num_pix, num_pix),
                                                     angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
            else:
                scalefactor = 1
                image = combined_map.reshape(num_pix, num_pix)

            labelfigs(ax[i], i, loc='tr')
            scalebar(ax[i],  plot_format['scalebar'][0] * scalefactor,
                     plot_format['scalebar'][1], loc='tr')

            # Plots the axes
            im = ax[i].imshow(image,
                              cmap=cmap_, clim=scales)

            colorbar(ax[i], im, ticks=False)

        else:

            # Plots the graphs
            hys_loop = ax[i].plot(
                voltage, clustered_ave_hysteresis[i - 1], cmap_.colors[i - 1])

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

            # Positions the binary image
            axes_in = plt.axes([pos.x0 + .06,
                                pos.y0 + .025,
                                .1, .1])
            combined_map.reshape(num_pix, num_pix)

            if plot_format['angle']:
                imageb, scalefactor = rotate_and_crop(clustered_maps[i - 1, :, :],
                                                      angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
            else:
                scalefactor = 1
                imageb = clustered_maps[i - 1, :, :]

            # Plots and formats the binary image
            axes_in.imshow(imageb,
                           cmap=cmap_2)
            axes_in.tick_params(
                axis='both', labelbottom=False, labelleft=False)

            labelfigs(ax[i], i, loc='br')

            set_axis_aspect(ax[i])

            savefig(folder + '/' + channel,
                    printing)


def training_loss(model_folder, data,
                  model, signal, signal_info,
                  printing, folder, i=None):
    """
    Plots the loss and reconstruction from neural network training

    Parameters
    ----------
    model_folder : string
        folder where the model is located
    data  : numpy array
        location where the data is saved
    model  : tensorflow object
        model for the trained neural network
    signal : string
        sets the signal to plot
    signal_info  : dict
        controls the design of the plots
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder  : string (optional)
        folder where the images will be saved
    i  : int (optional)
        manually selects the index to plot

    """

    voltage = data['raw']['voltage']
    if i is None:
        i = np.random.randint(0, data['normalized'][signal].shape[0])

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(3)

    # loads the text file and reorders the axes
    loss = np.rollaxis(np.loadtxt(model_folder + '/log.csv',
                                  skiprows=1, delimiter=','), 1)

    # plots the loss function
    ax[0].plot(loss[0],
               loss[1], 'r',
               loss[2], 'k')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')

    # plots the training data
    ax[1].plot(voltage, data['normalized'][signal][i], 'k')
    ax[1].plot(voltage, model.predict(np.atleast_3d(
        data['normalized'][signal][i])).squeeze(), 'r')

    # sets the axis titles
    ax[1].set_xlabel('Voltage (V)')
    ax[1].set_ylabel(signal_info[signal]['units'])
    ax[1].set_yticks(signal_info[signal]['y_tick'])
    ax[1].set_ylim([-2, 2])

    # resizes the figure
    set_axis_aspect(ax[1])

    # plots the validation data
    ax[2].plot(voltage, data['normalized']['val_' + signal][i], 'k')
    ax[2].plot(voltage,
               model.predict(np.atleast_3d(data['normalized']['val_' + signal][i])).squeeze(), 'r')

    # sets the axis titles
    ax[2].set_xlabel('Voltage (V)')
    ax[2].set_yticks(signal_info[signal]['y_tick'])
    ax[2].set_ylim([-2, 2])

    # resizes the figure
    set_axis_aspect(ax[2])

    plt.tight_layout(pad=0)

    # saves the figure
    savefig(folder + '/' + signal + '_{}'.format(i), printing)


def embedding_maps(data, printing, plot_format, folder, verbose=False, letter_labels=False,
                   filename='./embedding_maps', num_of_plots=True, ranges=None):
    """
    plots the embedding maps from a neural network

    Parameters
    ----------
    data : raw data to plot of embeddings
        data of embeddings
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    plot_format  : dict
        sets the plot format for the images
    folder : string
        set the folder where to export the images
    verbose : bool (optional)
        sets if the code should report information
    letter_labels : bool (optional)
        sets is labels should be included
    filename : string (optional)
        sets the filename for saving
    num_of_plots : int, optional
            number of principal components to show
    ranges : float, optional
            sets the clim of the images

    return
    ----------

    fig : object
        the figure pointer
    """

    # number of plots to show, if not provided shows all
    if num_of_plots:
        num_of_plots = data.shape[data.ndim - 1]

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(num_of_plots, mod=4)

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

    # plots all of the images
    for i in range(num_of_plots):
        if plot_format['rotation']:
            image, scalefactor = rotate_and_crop(data[:, i].reshape(original_size, original_size),
                                                 angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
        else:
            image = data[:, i].reshape(original_size, original_size)
            scalefactor = 1
        im = ax[i].imshow(image)
        ax[i].set_yticklabels('')
        ax[i].set_xticklabels('')

        if ranges is None:
            pass
        else:
            im.set_clim(0, ranges[i])

        # adds the colorbar
        if plot_format['color_bars']:
            colorbar(ax[i], im)

        # labels figures
        if letter_labels:
            labelfigs(ax[i], i)
        labelfigs(ax[i], i, string_add='emb. {0}'.format(i + 1), loc='bm')

        # adds the scalebar
        if plot_format['add_scalebar'] is not False:
            scalebar(ax[i], plot_format['scalebar'][0] * scalefactor,
                     plot_format['scalebar'][1])

    plt.tight_layout(pad=1)

    # saves the figure
    savefig(folder + '/' + filename, printing)

    return(fig)


def embedding_line_trace(ax,
                         map,
                         topo_map,
                         embedding_map,
                         climit,
                         plot_format,
                         resize_shape=900, number=3):
    """
    plots the embedding maps  with line trace from a neural network

    Parameters
    ----------
    ax : axis to plot
        data of embeddings
    map : list
        sets which map to plot
    embedding_map : float, array
        sets the embedding map to plot
    climit : float, array
        sets the colo range for the plot
    plot_format  : dict
        sets the plot format for the images
    resize_shape : int (optional)
        set the amount of the image to crop
    number : int (optional)
        sets the number of plots to show

    """
    # rotates and crops the image
    rotated_embedding, scale_factor = rotate_and_crop(
        embedding_map.reshape(60, 60),
        angle=plot_format['angle'],
        frac_rm=plot_format['frac_rm'])

    # resizes the image
    resize_embedding = resize(rotated_embedding, (resize_shape, resize_shape))

    # saves the current state of plot_format rotation
    hold = np.copy(plot_format['rotation'])

    # prevents the data from being rotated twice
    plot_format['rotation'] = False

    # plots the embedding map
    imagemap(ax[map + number],
             resize_embedding,
             plot_format,
             clim=climit)

    # sets the climit scale
    resize_embedding = set_value_scale(resize_embedding.reshape(-1),
                                       climit)
    # plots a line with a specific color map on top
    plot_line_with_color(ax[map], np.mean(topo_map, axis=0),
                         np.mean(resize_embedding.reshape(900, 900), axis=0), scale_factor)

    # restores the plot format variable
    plot_format['rotation'] = hold


def plot_line_with_color(ax, y, z, scale_factor):
    """
    plots a line with a colormap on top

    Parameters
    ----------
    ax : axis to plot
        data of embeddings
    y : float, array
        sets the line graph to plot
    z : float, array
        sets the embedding map to plot
    scale_factor : float
        sets the scaling for the scalebar size

    """
    # makes a linear spaced vector
    x = np.linspace(0, y.shape[0], y.shape[0])

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    lc = LineCollection(segments, cmap=plt.get_cmap('viridis'))
    lc.set_array(z)
    lc.set_linewidth(3)

    ax.add_collection(lc)

    # formats the figure
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-4e-9, 4e-9)
    ax.set_xticklabels(np.linspace(
        0, 500 * ((2000 * scale_factor) // 500), 5).astype(int))
    ax.set_xlabel('Position (nm)')
    ax.set_ylabel('Height (nm)')


def training_images(model,
                    data,
                    model_folder,
                    printing,
                    plot_format,
                    folder,
                    data_type='piezoresponse'):
    """
    plots the training images

    Parameters
    ----------
    model : tensorflow object
        neural network model
    data : float, array
        sets the line graph to plot
    model_folder : float, array
        sets the embedding map to plot
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    plot_format  : dict
        sets the plot format for the images
    folder : string
        set the folder where to export the images
    data_type : string (optional)
        sets the type of data which is used to construct the filename

    """

    # makes a copy of the format information to modify
    printing_ = printing.copy()
    plot_format_ = plot_format.copy()

    # sets to remove the color bars and not to print EPS
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

        # Computes the low dimensional layer
        embedding_exported[name_extraction(file_list)] = get_activations(model,
                                                                         data['normalized'][data_type], 9)

        # plots the embedding maps
        _ = embedding_maps(embedding_exported[name_extraction(file_list)], printing_, plot_format_,
                           folder,  filename='./epoch_{0:04}'.format(i))

        # Closes the figure
        plt.close(_)


def generator_movie(model,
                    encode,
                    voltage,
                    number,
                    averaging_number,
                    ranges,
                    folder,
                    plot_format,
                    printing,
                    graph_layout=[4, 4]):
    """
    plots the generator results

    Parameters
    ----------
    model : tensorflow object
        neural network model
    encode : float, array
        the input embedding (or output from encoder)
    voltage : float, array
        voltage array
    number : int
        number of divisions to plot
    averaging_number : int
        number of points to consider in the average
    ranges : float, array
        sets the ranges for the embeddings
    folder : string
        set the folder where to export the images
    plot_format  : dict
        sets the plot format for the images
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    graph_layout : int, array (optional)
        sets the layout for the figure.

    """

    # Defines the colorlist
    cmap = plt.get_cmap('viridis')

    # finds only those embeddings which are non-zero
    ind = np.where(np.mean(encode, axis=0) > 0)
    encode_small = encode[:, ind].squeeze()

    # pre-allocates the vector
    mean_loop = model.predict(np.atleast_2d(
        np.zeros((encode.shape[1])))).squeeze()

    for i in tqdm(range(number)):

        # builds the figure
        fig, ax = plt.subplots(graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0), graph_layout[1],
                               figsize=(3 * graph_layout[1], 3 * (graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0))))
        ax = ax.reshape(-1)

        # loops around all of the embeddings
        for j in range(len(ranges)):

            # linear space values for the embeddings
            value = np.linspace(0, ranges[j], number)

            # builds the embedding values for generation
            if i == 0:
                # all zero embeddings for the first index
                gen_value = np.zeros((encode.shape[1]))

            else:
                # finds the nearest point to the value and then takes the average
                # average number of points based on the averaging number
                idx = find_nearest(
                    encode_small[:, j], value[i], averaging_number)
                # computes the mean of the selected index
                gen_value = np.mean(encode[idx], axis=0)
                # specifically updates the value of the embedding to visualize based on the
                # linear spaced vector
                gen_value[j] = value[i]

            # generates the loop based on the model
            generated = model.predict(np.atleast_2d(gen_value)).squeeze()

            # plots the graph
            ax[j].plot(voltage, generated, color=cmap((i + 1) / number))

            # formats the graph
            ax[j].set_ylim(-2, 2)
            ax[j].set_yticklabels('')
            ax[j].plot(voltage, mean_loop, color=cmap((0 + 1) / number))
            ax[j].set_xlabel('Voltage (V)')

            # gets the position of the axis on the figure
            pos = ax[j].get_position()

            # plots and formats the binary cluster map
            axes_in = plt.axes([pos.x0 - .0105, pos.y0, .06 * 4, .06 * 4])

            # rotates the figure
            if plot_format['rotation']:
                imageb, scalefactor = rotate_and_crop(encode_small[:, j].reshape(60, 60),
                                                      angle=plot_format['angle'], frac_rm=plot_format['frac_rm'])
            else:
                scalefactor = 1
                imageb = encode_small[:, j].reshape(60, 60)

            # plots the imagemap and formats
            axes_in.imshow(imageb, clim=[0, ranges[j]])
            axes_in.set_yticklabels('')
            axes_in.set_xticklabels('')

        ax[0].set_ylabel('Piezoresponse (Arb. U.)')

        savefig(pjoin(folder, f'{i:04d}_maps'), printing)
        plt.close(fig)


def generator_piezoresponse(model,
                            embeddings,
                            voltage,
                            ranges,
                            number,
                            average_number,
                            printing,
                            plot_format,
                            folder,
                            y_scale=[-1.6, 1.6]):
    """
    plots the generator results

    Parameters
    ----------
    model : tensorflow object
        neural network model
    encode : float, array
        the input embedding (or output from encoder)
    voltage : float, array
        voltage array
    number : int
        number of divisions to plot
    averaging_number : int
        number of points to consider in the average
    ranges : float, array
        sets the ranges for the embeddings
    folder : string
        set the folder where to export the images
    plot_format  : dict
        sets the plot format for the images
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    graph_layout : int, array (optional)
        sets the layout for the figure.

    """

    # sets the colormap
    cmap = plt.cm.viridis

    # finds places where the embedding map has non-zero index
    ind = np.where(np.mean(embeddings, axis=0) > 0)
    embedding_small = embeddings[:, ind].squeeze()

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(embedding_small.shape[1] * 2)

    # plots all of the embedding maps
    for i in range(embedding_small.shape[1]):

        im = imagemap(ax[i], embedding_small[:, i].reshape(60, 60),
                      plot_format, clim=[0, ranges[i]])

    # loops around the number of example loops
    for i in range(number):

        # loops around the number of embeddings from the range file
        for j in range(len(ranges)):

            # sets the linear spaced values
            value = np.linspace(0, ranges[j], number)

            # builds the embedding vector. First is all 0's
            if i == 0:
                gen_value = np.zeros((embeddings.shape[1]))
            else:
                idx = find_nearest(
                    embedding_small[:, j], value[i], average_number)
                gen_value = np.mean(embeddings[idx], axis=0)
                gen_value[j] = value[i]

            # computes the generated results
            generated = model.predict(np.atleast_2d(gen_value)).squeeze()

            # plots and formats the graphs
            ax[j + embedding_small.shape[1]
               ].plot(voltage, generated, color=cmap((i + 1) / number))
            ax[j + embedding_small.shape[1]].set_ylim(y_scale)
            ax[j + embedding_small.shape[1]].set_yticklabels('')
            ax[j + embedding_small.shape[1]].set_xlabel('Voltage (V)')
            if j == 0:
                ax[j + embedding_small.shape[1]
                   ].set_ylabel('Piezoresponse (Arb. U.)')
            plt.tight_layout(pad=1)

    # saves the figure
    savefig(folder + '/generated_loops',  printing)


def resonance_generator_movie(
        model,
        index_c,
        index_a,
        embedding_c,
        voltage,
        embedding_a,
        ranges_c,
        ranges_a,
        number,
        averaging_number,
        resonance_decode,
        plot_format,
        printing,
        folder,
        graph_layout=[6, 4]):
    """
    plots the generator results for the resonance frequency

    Parameters
    ----------
    model : tensorflow object
        neural network model
    index_c : int, list
        the index for the embeddings in the c domains
    index_a : int, list
        the index for the embeddings in the a domains
    embedding_c : float, array
        embedding map for the c domain
    voltage : float, array
        voltage vector
    embedding_a : float, array
        embedding map for the a domain
    ranges_c : float, list
        ranges for the c domains
    ranges_a : float, list
        ranges for the a domains
    number : int
        number of divisions to plot
    averaging_number : int
        number of points to consider in the average
    resonance decode : tensorflow object
        neural network decoder model
    plot_format  : dict
        sets the plot format for the images
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        set the folder where to export the images
    graph_layout : int, array (optional)
        sets the layout for the figure.

    """

    # Defines the colorlist
    cmap = plt.get_cmap('viridis')

    # Loop for each of the example graphs
    for i in tqdm(range(number)):

        # builds the figure
        fig, ax = plt.subplots(graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0), graph_layout[1],
                               figsize=(3 * graph_layout[1], 3 * (graph_layout[0] // graph_layout[1] + (graph_layout[0] % graph_layout[1] > 0))))
        ax = ax.reshape(-1)

        # makes a single plot of a results
        single_resonance_generator(
            resonance_decode,
            ax[:graph_layout[0] // 2],
            i,
            index_a,
            embedding_a,
            ranges_a,
            averaging_number,
            number,
            voltage,
            plot_format)

        # makes a single plot of a results
        single_resonance_generator(
            resonance_decode,
            ax[graph_layout[0] // 2:],
            i,
            index_c,
            embedding_c,
            ranges_c,
            averaging_number,
            number, voltage,
            plot_format)

        # saves the figure
        savefig(pjoin(folder, f"{i:04d}_maps"), printing)
        plt.close(fig)


def single_resonance_generator(
        model,
        ax,
        i,
        index,
        embedding,
        ranges,
        averaging_number,
        number,
        voltage,
        plot_format):
    """
    plots a single plot of a resonance results

    Parameters
    ----------
    model : tensorflow object
        neural network model
    ax : object
        pointer to an axis
    i : int
        sets the graph number
    index : int, list
        the index for the embeddings
    embedding : float, array
        embedding map
    ranges : float, list
        ranges for the embedding
    averaging_number : int
        number of points to consider in the average
    number : int
        number of divisions to plot
    voltage : float, array
        voltage vector
    plot_format  : dict
        sets the plot format for the images

    """
    # loops around the embeddings
    for j, index_ in enumerate(index):

        # plots the imagemap
        im = imagemap(ax[j], embedding[:, index_], plot_format,
                      clim=[0, ranges[index_]])

        # computes the values for the generator
        value = np.linspace(0, ranges[index_], number)

        # generates the loop for the average loop
        mean_loop = model.predict(np.atleast_2d(
            np.zeros((len(ranges))))).squeeze()

        # builds the embedding vectors for generation
        if i == 0:
            gen_value = np.zeros((len(ranges)))
        else:
            idx = find_nearest(
                embedding[:, index_], value[i], averaging_number)
            gen_value = np.mean(embedding[idx], axis=0)
            gen_value[index_] = value[i]

        # computes the generator curves
        generated = model.predict(np.atleast_2d(gen_value)).squeeze()

        # plots and formats the graph
        ax[j + len(index)].plot(voltage, generated,
                                color=cmap((i + 1) / number))
        ax[j + len(index)].set_ylim(-2, 2)
        ax[j + len(index)].set_yticklabels('')
        ax[j + len(index)].plot(voltage, mean_loop,
                                color=cmap((0 + 1) / number))


def resonance_generator(
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
        name_prefix=''):
    """
    plots the generator results for the resonance frequency

    Parameters
    ----------
    model : tensorflow object
        neural network model (resonance)
    model_piezo : tensorflow object
        neural network model (piezoresponse)
    index : int, list
        the index for the embeddings in the c domains
    embedding : float, array
        embedding map (resonance)
    ranges : float, list
        ranges for the embedding map
    number : int
        number of divisions to plot
    averaging_number : int
        number of points to consider in the average
    plot_subselect : int, list
        selects the curves and the plots to show
    embedding_piezo : float, array
        embedding map (piezoresponse)
    voltage : float, array
        voltage vector
    resonance_cleaned : float, array
        cleaned resonance data
    plot_format  : dict
        sets the plot format for the images
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    folder : string
        set the folder where to export the images
    scales : float, list
        sets the y limits of the plots
    name_prefix : string
        prefix to add to the file prior to saving

    """

    # selects the colormap
    cmap = plt.cm.viridis

    # defines the colorlist
    cmap2 = plt.get_cmap('plasma')

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(len(index) * 3, mod=len(index))

    # shifts the voltage
    shift_voltage = roll_and_append(voltage)

    # loops around the embedding maps to plot
    for i, index_ in enumerate(index):

        # plots the image map of the region
        im = imagemap(ax[i],
                      embedding[:, index_],
                      plot_format,
                      clim=[0, ranges[index_]])

    # loops around the number of example loops
    for i in range(number):

        # loops around the selected index
        for j, index_ in enumerate(index):

            # builds a linear space vector where we tune the embedding value
            value = np.linspace(0, ranges[index_], number)

            # plots the average loop (all embedding = 0)
            if i == 0:
                gen_value = np.zeros((len(ranges)))
            else:
                # finds a select number of indices with a value closest to the selected value
                idx = find_nearest(
                    embedding[:, index_], value[i], averaging_number)

                # computes the mean
                gen_value = np.mean(embedding[idx], axis=0)

                # replaces the value with the selected value
                gen_value[index_] = value[i]

                # finds the embedding of the piezoelectric loop (finds points closest to the average)
                gen_value_piezo = np.mean(embedding_piezo[idx], axis=0)

            # plots only those graphs selected
            if i in plot_subselect[j]:

                # generates the curves
                generated = model.predict(np.atleast_2d(gen_value)).squeeze()
                generated_piezo = model_piezo.predict(
                    np.atleast_2d(gen_value_piezo)).squeeze()

                # connects the first and last point
                generated = roll_and_append(generated)

                # rescales the data back to the original data space
                generated *= np.std(resonance_cleaned.reshape(-1))
                generated += np.mean(resonance_cleaned.reshape(-1))

                # plots the resonance curves
                ax[j + len(index)].plot(shift_voltage, generated,
                                        color=cmap((i + 1) / number), linewidth=3)

                # plots the piezoresponse curves
                ax[j + len(index) * 2].plot(voltage, generated_piezo,
                                            color=cmap((i + 1) / number), linewidth=3)

                # formats the plots
                if j == 0:
                    ax[j + len(index)].set_ylabel('Resonance (kHz)')
                    ax[j + len(index) * 2].set_ylabel('Piezoresponse (Arb. U.)')
                else:
                    ax[j + len(index)].set_yticklabels('')

                ax[j + len(index) * 2].set_yticklabels('')

                ax[j + len(index)].set_xticklabels('')
                ax[j + len(index) * 2].set_xlabel('Voltage (V)')

                # sets the scales
                if scales is None:
                    pass
                else:
                    ax[j + len(index)].set_ylim(scales[0])
                    ax[j + len(index) * 2].set_ylim(scales[1])

    plt.tight_layout(pad=0)

    # saves the figure
    savefig(folder + '/' +
            name_prefix +
            '_generated_autoencoded_result',
            printing)
