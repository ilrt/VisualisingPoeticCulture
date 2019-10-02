"""
Visualising Poetry module
"""

import glob
import numpy as np
import os
import pandas as pd
import settings as settings
import shutil
import urllib
from zipfile import ZipFile

import private_settings as private

# ---------- Constants (Excel and Data Frame headers)

POEM_DATA_SHEET = 'poem data'
PUB_TITLE = 'publication title'
PUB_TYPE = 'publication type'
REF_NO = 'ref no'
YEAR = 'year'
MONTH = 'month'
DAY = 'day'
F_LINE = 'first line'


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


def clean_all():
    """ Clean the environment """
    clean_preprocessed()
    clean_source()


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
    """
    Turn the Excel spreadsheet into a Pandas DataFrame and create a pickle file.
    We 'clean' the data before creating the pickle file.
    """

    # create the pickles directory if it doesn't exist
    if not os.path.exists(settings.PICKLE_SRC):
        os.makedirs(settings.PICKLE_SRC)

    # go through sub folders and get the full name of Excel (.xlsx) files
    for file in glob.glob(settings.DATA_SRC + '**/*.xlsx', recursive=True):
        # get the filename (without folders and file extension)
        filename = file.split("/")[-1]
        filename = filename.replace('.xlsx', '')

        # open the excel file and get the poem data
        df = pd.read_excel(file, sheet_name=POEM_DATA_SHEET)

        # drop empty rows
        df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)

        # make sure the year is int64
        df[YEAR] = df[YEAR].astype(np.int64)

        # make sure a ref no is an int
        df[REF_NO] = df[REF_NO].astype(pd.Int64Dtype())

        # strip whitespace around publication title
        df[PUB_TITLE] = df[PUB_TITLE].str.strip()

        # strip whitespace and normalize case on month
        df[MONTH] = df[MONTH].str.strip()
        df[MONTH] = df[MONTH].str.title()

        # consistent case for publication type
        df[PUB_TYPE] = df[PUB_TYPE].str.lower()

        # write the pickle file
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


# ---------- Helper methods

def start_year(df):
    """ Get the earliest year in a data frame """
    return df[YEAR].min()


def end_year(df):
    """ Get the latest year in a data frame """
    return df[YEAR].max()


def create_publication_year_matrix(df):
    """ Create a matrix of publications and years so we can see the coverage in a data frame """

    # we'll have titles as the index and years as the columns
    year_range_column = np.arange(start_year(df), end_year(df) + 1, 1)
    pub_title_index = df[PUB_TITLE].unique()

    # create a matrix of titles and possible years with a value of zero
    matrix = pd.DataFrame(np.zeros(shape=(pub_title_index.size, year_range_column.size)), columns=year_range_column,
                          index=pub_title_index)

    # Add 1 to a title / year combination
    for title in pub_title_index:
        sub_df = df[df[PUB_TITLE] == title]
        sub_years = sub_df[YEAR].unique()
        for year in sub_years:
            matrix.at[title, year] += 1

    return matrix


def create_publications_matrix(df):
    """" Create a matrix of publications where the score is incremented when a poem is shared
         between publications. Reprints in the same publication are not included. """

    # publications (ignore duplicates)
    pubs = df[PUB_TITLE].unique()

    # create a matrix of publications with a value of zero
    matrix_df = pd.DataFrame(np.zeros(shape=(pubs.size, pubs.size)), columns=pubs, index=pubs)

    # group by ref number (i.e the same poem)
    grouped = df.groupby(REF_NO)

    # go through the groups and update matrix
    for group_name, df_group in grouped:
        # go through the publication on one axis
        for pub_x in df_group[PUB_TITLE].unique():
            # go through the group on the other axes
            for pub_y in df_group[PUB_TITLE].unique():
                # increment score if the publications don't match
                if pub_y != pub_x:
                    matrix_df.at[pub_x, pub_y] += 1

    return matrix_df
