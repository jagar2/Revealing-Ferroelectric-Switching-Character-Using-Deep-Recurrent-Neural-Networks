"""
Created on Tue Oct 09 16:39:00 2018
@author: Joshua C. Agar
"""

import sys
import time
import urllib
import zipfile
import shutil
import os.path
import os
import numpy as np


def reporthook(count, block_size, total_size):
    """
    Function that displays the status and speed of the download

    """

    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def download_file(url, filename):
    """
    Function that downloads the data file from a URL

    Parameters
    ----------

    url : string
        url where the file to download is located
    filename : string
        location where to save the file
    reporthook : function
        callback to display the download progress

    """
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename, reporthook)


def compress_folder(base_name, format, root_dir=None):
    """
    Function that zips a folder can save zip and tar

    Parameters
    ----------

    base_name : string
        base name of the zip file
    format : string
        sets the format of the zip file. Can either be zip or tar
    root_dir : string (optional)
        sets the root directory to save the file

    """

    shutil.make_archive(base_name, format, root_dir)

def unzip(filename, path):
    """
    Function that unzips the files

    Parameters
    ----------

    filename : string
        base name of the zip file
    path : string
        path where the zip file will be saved

    """
    zip_ref = zipfile.ZipFile('./' + filename, 'r')
    zip_ref.extractall(path)
    zip_ref.close()

def get_size(start_path='.'):
    """

    Function that computes the size of a folder

   Parameters
   ----------

   start_path : string
       Path to compute the size of

    Return
   ----------

   total_size : float
       Size of the folder
    """

    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def download_and_unzip(filename, url, save_path, download_data):
    """

        Function that computes the size of a folder

       Parameters
       ----------

       filename : string
           filename to save the zip file
        url : string
           url where the file is located
        save_path : string
           place where the data is saved
        download_data : bool
           sets if to download the data

    """
    if np.int(get_size(save_path) / 1e9) < 1:
        if np.int(get_size(save_path) / 1e9) > 1:
            print('Using files already downloaded')
        elif download_data:
            print('downloading data')
            download_file(url, filename)
        elif os.path.isfile(filename):
            print('Using zip file already available')
        else:
            pass

        if os.path.isfile(filename):
            print(f'extracting {filename} to {save_path}')
            unzip(filename, save_path)
