"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

from .format import *
import numpy as np
from ..util.file import *
from ..util.core import *
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob


# Defines a set of custom color maps
color1 = plt.cm.viridis(np.linspace(.5, 1, 128))
color2 = plt.cm.plasma(np.linspace(1, .5, 128))


mymap = LinearSegmentedColormap.from_list(
    'my_colormap', np.vstack((color1, color2)))


def phase_field_switching(phase_field_information,
                          printing):
    """
    plots the phase field switching

    Parameters
    ----------
    phase_field_information : dict
        all information for plotting
        tips :  string, list
            name of the tips and folder locations
       folder : dict
            locations of all the folders
                time_series : string
                    folder where the time series is located
                polarization : string
                    folder where the polarization is located
                energy : string
                    folder where the energy files are located
         time_step : int, list
            timesteps to plot
        tip_positions : dict, int
        graph_info : dict
            information for the format of the plot
        labels : string, list
            labels for the figures
        output_folder : string
            folder where to output the files
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG

    """

    # preallocate the axis
    ax = [None]*6*3

    # loops around the number of tips
    for tip in phase_field_information['tips']:

        # makes the figures
        fig = plt.figure(figsize=(6, 2*6))

        # builds the axes for the figure
        for j in range(6):

            for i in range(3):

                # builds the figure
                ax[i+(j*3)] = plt.subplot2grid((6, 6),
                                               (j, i*2),
                                               colspan=2,
                                               rowspan=1)

        timedata = np.genfromtxt(phase_field_information['folder']['time_series'] +
                                 tip + '/timedata.dat')
        files = np.sort(glob.glob(phase_field_information['folder']['polarization'] +
                                  'data/' + tip + '/*.dat'))
        files_energy = np.sort(glob.glob(phase_field_information['folder']['energy'] +
                                         tip + '/*.dat'))

        # loops around the selected timesteps
        for i, index in enumerate(phase_field_information['time_step']):

            # gets the tip position
            pos = phase_field_information['tip_positions'][tip]['pos']

            # reads the data
            data = np.genfromtxt(files[index], skip_header=1, skip_footer=3)
            data_energy = np.genfromtxt(
                files_energy[index], skip_header=1, skip_footer=3)

            # find the points where we want to plot
            idz = np.where(
                data[:, 2] == phase_field_information['graph_info']['top'])
            idy = np.where(
                data[:, 1] == phase_field_information['graph_info']['y_cut'])

            # extracts the desired limits
            ylim = phase_field_information['graph_info']['y_lim']
            xlim = phase_field_information['graph_info']['x_lim']

            topodata = np.rollaxis(data[idy, 2+3].reshape(128, -1), 1)

            # loops around each signal type
            for k, labels in enumerate(phase_field_information['labels']):

                if labels == 'Polarization Z':
                    data_pz = np.rollaxis(data[idy, 5].reshape(128, -1), 1)
                    # gives the plot 3-dimensions
                    data_out = impose_topography(topodata, data_pz)
                else:
                    data_energy_ = np.rollaxis(
                        data_energy[idy, 2+k].reshape(128, -1), 1)
                    # gives the plot 3-dimensions
                    data_out = impose_topography(topodata, data_energy_)

                # plots the phase field results
                im = ax[i+(3*k)].imshow(data_out,
                                        clim=phase_field_information['graph_info']['clim'][labels],
                                        cmap=mymap)

                # sets the x and y limits
                ax[i+(3*k)].set_ylim(ylim[0], ylim[1])
                ax[i+(3*k)].set_xlim(xlim[0], xlim[1])

                # defines the tip height based on the highest value
                tip_height = np.max(np.argwhere(
                    np.isfinite(data_out[:, pos[0]*4])))

                # adds an afm tip to the image
                ax[i+(3*k)].annotate('', xy=(pos[0]*4, tip_height),
                                     xycoords='data',
                                     xytext=(pos[0]*4, tip_height+20),
                                     textcoords='data',
                                     arrowprops=dict(arrowstyle="wedge", facecolor='k'))

                # formats the figures
                if i == 1:
                    # adds a label
                    ax[i+(3*k)].set_title(labels)
                elif i == 2:
                    # adds a colorbar
                    divider = make_axes_locatable(ax[i+(3*k)])
                    cax = divider.append_axes('right',
                                              size='10%',
                                              pad=0.05)
                    cbar = plt.colorbar(im, cax=cax)
                    tick_locator = ticker.MaxNLocator(nbins=3)
                    cbar.locator = tick_locator
                    cbar.update_ticks()

                    # adds a label for the colorbar
                    if k != 0:
                        cbar.set_label('$J/m^{3}$',
                                       rotation=270,
                                       labelpad=6)

        for axes in ax:
            # Removes axis labels
            axes.set_xticks([])
            axes.set_yticks([])
            axes.axison = False

        # saves the figure
        savefig(phase_field_information['output_folder'] +
                '/' + tip + '_x_{}_y_{}'.format(pos[0], pos[1]),
                printing)


def impose_topography(topodata,
                      data):
    """
    adds topography to phase field data

    Parameters
    ----------
    topodata : float, array
        the topography array calculated
    data : float, array
        data to plot

    Return
    ----------
    out_image : float, array
        output image after imposing topography

    """

    # builds the empty matrix
    out_image = np.empty((int(20*5)+20,
                          128*4))
    out_image[:, :] = np.nan

    # loops around the x axis
    for j in range(topodata.shape[1]):

        # resets the count for each column
        count = 0

        for i in range(topodata.shape[0]):

            # if the polarization is >0.5 make 5 pixels. If not make 4.
            if np.abs(topodata[i, j]) > .5:
                out_image[count:count+5, j*4:j*4+4] = data[i, j]
                count += 5
            else:
                out_image[count:count+4, j*4:j*4+4] = data[i, j]
                count += 4

    return out_image


def movie(phase_field_information,
          printing):
    """
    plots the phase field switching

    Parameters
    ----------
    phase_field_information : dict
        all information for plotting
        tips :  string, list
            name of the tips and folder locations
       folder : dict
            locations of all the folders
                time_series : string
                    folder where the time series is located
                polarization : string
                    folder where the polarization is located
                energy : string
                    folder where the energy files are located
         time_step : int, list
            timesteps to plot
        tip_positions : dict, int
        graph_info : dict
            information for the format of the plot
        labels : string, list
            labels for the figures
        output_folder : string
            folder where to output the files
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG

    """

    # loops around the  tips available
    for tip in phase_field_information['tips']:

        # lays out the figure
        fig, ax = layout_fig(8, 4)

        # reads the data file
        timedata = np.genfromtxt(phase_field_information['folder']['time_series'] +
                                 tip + '/timedata.dat')
        files = np.sort(glob.glob(phase_field_information['folder']['polarization'] +
                                  'data/' + tip + '/*.dat'))
        files_energy = np.sort(glob.glob(phase_field_information['folder']['energy'] +
                                         tip + '/*.dat'))

        # extracts only some of the points where images are exported
        phase_field_voltage = timedata[::5, 2]

        # builds an example loop based on the loop fitting function
        fit_results = loop_fitting_function(phase_field_voltage,
                                            -1, 1, 0, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 0, 0, -5, 5)
        example_loop = np.concatenate((fit_results['Branch2'][0:20],
                                       fit_results['Branch1'][20:60],
                                       fit_results['Branch2'][60::]))

        # makes folder to export files to
        current_folder = make_folder(phase_field_information['output_folder'] +
                                     '/movie/' + tip)

        # loops around the number of files
        for index in range(files.shape[0]):

            fig, ax = layout_fig(8, 4)

            # gets the tip position
            pos = phase_field_information['tip_positions'][tip]['pos']

            # loads the data
            data = np.genfromtxt(files[index], skip_header=1,
                                 skip_footer=3)
            data_energy = np.genfromtxt(files_energy[index],
                                        skip_header=1,
                                        skip_footer=3)

            # finds and slices the data where selected
            idz = np.where(
                data[:, 2] == phase_field_information['graph_info']['top'])
            idy = np.where(
                data[:, 1] == phase_field_information['graph_info']['y_cut'])
            ylim = phase_field_information['graph_info']['y_lim']
            xlim = phase_field_information['graph_info']['x_lim']

            # extracts the data for the topography
            topodata = np.rollaxis(data[idy, 2+3].reshape(128, -1), 1)

            for k, labels in enumerate(phase_field_information['labels']):

                if k > 2:
                    inds = k+2
                else:
                    inds = k+1

                # imposes the topography on the figure
                if labels == 'Polarization Z':
                    data_pz = np.rollaxis(data[idy, 5].reshape(128, -1), 1)
                    data_out = impose_topography(topodata, data_pz)
                else:
                    data_energy_ = np.rollaxis(
                        data_energy[idy, 2+k].reshape(128, -1), 1)
                    data_out = impose_topography(topodata, data_energy_)

                # shows the figure
                im = ax[inds].imshow(data_out,
                                     clim=phase_field_information['graph_info']['clim'][labels],
                                     cmap=mymap)

                # sets the limits
                ax[inds].set_ylim(ylim[0], ylim[1])
                ax[inds].set_xlim(xlim[0], xlim[1])

                # defines the tip height based on the highest value
                tip_height = np.max(np.argwhere(
                    np.isfinite(data_out[:, pos[0]*4])))

                # adds a tip to the image
                ax[inds].annotate('', xy=(pos[0]*4, tip_height),
                                  xycoords='data',
                                  xytext=(pos[0]*4, tip_height+20),
                                  textcoords='data',
                                  arrowprops=dict(arrowstyle="wedge", facecolor='k'))

                # adds a title
                ax[inds].set_title(labels)

            # formats the plot
            ax[0].plot(phase_field_voltage, example_loop)
            ax[0].plot(phase_field_voltage[index], example_loop[index], 'o')
            ax[0].set_xlabel('Voltage')
            ax[0].set_ylabel('Polarization')
            ax[0].set_yticks([])

            ax[4].plot(phase_field_voltage)
            ax[4].plot(index, phase_field_voltage[index], 'o')
            ax[4].set_xlabel('Time Step')
            ax[4].set_ylabel('Voltage (V)')

            for j, axes in enumerate(ax):

                if j not in [0, 4]:
                    # Removes axis labels
                    axes.set_xticks([])
                    axes.set_yticks([])
                    axes.axison = False

            # saves the figure
            savefig(current_folder + '/Image_{0:03d}'.format(index+1),
                    printing)

            # closes the figure after exporting
            plt.close(fig)


def phase_field_hysteresis(phase_field_information, printing):
    """
    plots the hysteresis loop from phase field simulations

    Parameters
    ----------
    phase_field_information : dict
        all information for plotting
        tips :  string, list
            name of the tips and folder locations
       folder : dict
            locations of all the folders
                time_series : string
                    folder where the time series is located
                polarization : string
                    folder where the polarization is located
                energy : string
                    folder where the energy files are located
         time_step : int, list
            timesteps to plot
        tip_positions : dict, int
        graph_info : dict
            information for the format of the plot
        labels : string, list
            labels for the figures
        output_folder : string
            folder where to output the files
    printing : dictionary
        contains information for printing
        'dpi': int
            resolution of exported image
        print_EPS : bool
            selects if export the EPS
        print_PNG : bool
            selects if print the PNG
    """

    index = 0

    fig, ax = layout_fig(10, mod=5)

    # loops around the number of tips
    for i, tip in enumerate(phase_field_information['tips']):
        timedata = np.genfromtxt(phase_field_information['folder']['time_series'] +
                                 tip + '/timedata.dat')
        files = np.sort(glob.glob(
            phase_field_information['folder']['polarization'] + 'data/' + tip + '/*.dat'))
        files_energy = np.sort(glob.glob(phase_field_information['folder']['energy'] +
                                         tip + '/*.dat'))

        # gets the tip position
        pos = phase_field_information['tip_positions'][tip]['pos']

        # reads the data
        data = np.genfromtxt(files[index], skip_header=1, skip_footer=3)

        # find the points where we want to plot
        idz = np.where(
            data[:, 2] == phase_field_information['graph_info']['top'])
        idy = np.where(
            data[:, 1] == phase_field_information['graph_info']['y_cut'])

        # extracts the desired limits
        ylim = phase_field_information['graph_info']['y_lim']
        xlim = phase_field_information['graph_info']['x_lim']

        topodata = np.rollaxis(data[idy, 2 + 3].reshape(128, -1), 1)

        data_pz = np.rollaxis(data[idy, 5].reshape(128, -1), 1)
        # gives the plot 3-dimensions
        data_out = impose_topography(topodata, data_pz)

        # plots the phase field results
        im = ax[i].imshow(data_out,
                          clim=phase_field_information['graph_info']['clim']['Polarization Z'],
                          cmap=mymap)

        # sets the x and y limits
        ax[i].set_ylim(ylim[0], ylim[1])
        ax[i].set_xlim(xlim[0], xlim[1])

        # defines the tip height based on the highest value
        tip_height = np.max(np.argwhere(
            np.isfinite(data_out[:, pos[0] * 4])))

        # adds an afm tip to the image
        ax[i].annotate('', xy=(pos[0] * 4, tip_height),
                       xycoords='data',
                       xytext=(pos[0] * 4, tip_height + 20),
                       textcoords='data',
                       arrowprops=dict(arrowstyle="wedge", facecolor='k'))

        # Removes axis labels
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].axison = False

        ax[i + 5].plot(timedata[:, 2] * -1, timedata[:, 3], 'k')
        # Removes axis labels
        ax[i + 5].set_ylim([-1.2, 1.2])
        ax[i + 5].set_xlabel('Voltage (V)')
        if i == 0:
            ax[i + 5].set_ylabel('Polarization (Arb. U.)')

        plt.tight_layout()

    # saves the figure
    savefig(phase_field_information['output_folder'] +
            '/' + 'Phase_field_loops',
            printing)
