"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

import numpy as np
from ..util.core import *
from sklearn import (decomposition, preprocessing as pre, cluster)


def pca(loops, n_components=15, verbose=True):
    """
    Computes the PCA and forms a low-rank representation of a series of response curves
    This code can be applied to all forms of response curves.
    loops = [number of samples, response spectra for each sample]

    Parameters
    ----------
    loops : numpy array
        1 or 2d numpy array - [number of samples, response spectra for each sample]
    n_components : int, optional
        int - sets the number of components to save
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
        PCA_reconstructed = PCA_reconstructed.reshape(
            original_size, original_size, -1)
    except:
        pass

    return PCA, PCA_reconstructed


def weights_as_embeddings(pca, loops, num_of_components=0, verbose=True):
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

    return PCA_embedding


def nmf(model, data):
    """
    Computes the nmf

    Parameters
    ----------
    model : object
        object of the nmf computation
    data: numpy, array
        raw data to conduct nmf
    num_of _components: int
        number of PCA components to compute

    Returns
    -------
    W : object
        nmf fit results
    H : numpy array
        components of nmf
    """

    # Fits the NMF model
    W = model.fit_transform(np.rollaxis(data - np.min(data), 1))
    H = np.rollaxis(model.components_, 1)

    return W, H


def k_means_clustering(input_data, channel, clustering, seed=[], pca_in=True):
    """
    Clusters the loops

    Parameters
    ----------
    input_data : float
        data for clustering
    channel : string
        data channel for clustering
    clustering : dict
        number of clusters for each type
    seed : int
        random seed for regular clustering
    pca_in : object (optional)
        pca data for clustering

    """

    if pca_in is True:
        data = input_data['sg_filtered'][channel]
    else:
        data = pca_in

    data_piezo = input_data['sg_filtered']['piezoresponse']

    if seed != []:
        # Defines the random seed for consistent clustering
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

    # Enumerates the k-means clustering map for plotting
    combined_map = a_map + c_map

    return combined_map, cluster_ca, c_map, a_map
