""" Settings file dealing with directory locations etc."""

import os


DATA_ROOT = os.getcwd().replace('notebooks', '')

# location of the Excel files
DATA_SRC = DATA_ROOT + '/data/source/'

# location of pickled Pandas data frames
PICKLE_SRC = DATA_ROOT + '/data/preprocessed/'

# name the file we download from DropBox
DB_FILE = 'database.zip'
