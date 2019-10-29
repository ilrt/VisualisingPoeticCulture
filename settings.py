""" Settings file dealing with directory locations etc."""

import os


DATA_ROOT = os.getcwd().replace('notebooks', '')

# location of the Excel files
DATA_SRC = DATA_ROOT + '/data/source/'

# location of pickled Pandas data frames
PICKLE_SRC = DATA_ROOT + '/data/preprocessed/'

# location to puts reports
REPORTS_DIR = DATA_ROOT + '/reports/'

# name the file we download from DropBox
DB_FILE = 'database.zip'

# We currently have an outlier of 1792 in the dataset, with nothing in between 1760
# and 1792. We cab use this to filter out any years beyond the value set.
MAX_YEAR = 1760
