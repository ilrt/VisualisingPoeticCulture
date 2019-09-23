"""
Visualising Poetry module
"""

import glob
import pandas as pd
import settings as settings
from zipfile import ZipFile
import shutil
import os
import urllib
import private_settings as private


# ---------- Methods for cleaning the data folder

def clean(target):
    """ Clean a target directory of files and sub directories """
    if os.path.exists(target):
        shutil.rmtree(target)
    os.makedirs(target)


def clean_source():
    """ Clean the source directory of Excel files """
    clean(settings.DATA_SRC)


def clean_preprocessed():
    """ Clean the preprocessed directory """
    clean(settings.PICKLE_SRC)


# ---------- Methods for getting and preprocessing data

def download_data():
    """ Get the zip file specified in the private settings """
    urllib.request.urlretrieve(private.DATABASE_ZIP_URL, settings.DATA_SRC + settings.DB_FILE)


def unpack_zip():
    """ Extract a zip file of Excel files """
    for file in glob.glob(settings.DATA_SRC + '*.zip', recursive=False):
        with ZipFile(file) as zip_object:
            for item in zip_object.namelist():
                if item.endswith('.xlsx'):
                    zip_object.extract(item, path=settings.DATA_SRC)


def write_pickle_data_frames():
    """ Create a data frame and pickle of the Excel sheet with poem data """
    for file in glob.glob(settings.DATA_SRC + '**/*.xlsx', recursive=True):
        filename = file.split("/")[-1]
        filename = filename.replace('.xlsx', '')
        df = pd.read_excel(file, sheet_name='poem data')
        df.to_pickle(settings.PICKLE_SRC + filename + '.pickle')


# ---------- Methods for setting an environment up for a notebook

def setup():
    """ Setup the environment so the notebooks have data. """
    # clean directories
    clean_source()
    clean_preprocessed()
    # download the data
    download_data()
    unpack_zip()
    # create panda data frames and pickle them
    write_pickle_data_frames()


def setup_if_needed():
    """ Helper method to run at the start of a notebook to ensure that there is data """
    if not os.path.exists(settings.DATA_SRC) or not os.path.exists(settings.DATA_SRC + settings.DB_FILE):
        print("No zip file. Getting data.")
        setup()
        print("Setup complete")


def pickle_as_single_data_frame():
    """ Load all the pickle files as a single data frame """
    files = glob.glob(settings.PICKLE_SRC + '*.pickle', recursive=True)
    results = []
    for file in files:
        results.append(pd.read_pickle(file))

    return pd.concat(results, sort=False)
