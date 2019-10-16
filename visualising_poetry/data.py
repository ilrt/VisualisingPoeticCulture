"""
Visualising Poetry module
"""

import glob
import re

import numpy as np
import os
import pandas as pd
from IPython.core.display import display, HTML
import matplotlib.pyplot as plot
import seaborn as sn

import settings as settings
import shutil
import urllib
from zipfile import ZipFile
import ipywidgets as widgets

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
ATTR_TYPE = 'attribution type'

# computed
PRINTED_DATE = 'printed'
PRINTED_DATE_STR = 'printed (string)'

# used to convert strings used in the dataset to numbers
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12, 'Dec Supp': 12, 'Prefatory': 1
}

attr_type_regex = re.compile('^[m|f]\. pseud$')


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

        # make sure the day is int64
        df[DAY] = df[DAY].astype(pd.Int64Dtype())

        # make sure a ref no is an int
        df[REF_NO] = df[REF_NO].astype(pd.Int64Dtype())

        # strip whitespace around publication title
        df[PUB_TITLE] = df[PUB_TITLE].str.strip()

        # strip whitespace and normalize case on month
        df[MONTH] = df[MONTH].str.strip()
        df[MONTH] = df[MONTH].str.title()

        # consistent case for publication type
        df[PUB_TYPE] = df[PUB_TYPE].str.lower()

        # create a datetime object for the calculated printed date
        df[PRINTED_DATE] = df.apply(create_printed_datetime, axis=1)

        # Create a string representation ... in case we want to view in Excel
        df[PRINTED_DATE_STR] = df.apply(date_to_string, axis=1)

        # clean the attribution type
        df[ATTR_TYPE] = df.apply(clean_attribution_type, axis=1)

        # write the pickle file
        df.to_pickle(settings.PICKLE_SRC + filename + '.pickle')


# ---------- Data cleansing methods

def create_printed_datetime(row):
    """
    Create a calculated numpy datetime64 object to represent the estimated date the publication was  printed and/or
    distributed. If the year, month and day are given we use that. If no day is given we set the value to the 1st of
    the following month, since these publications were usually distributed the start of the month following the month
    given on the publication. 'Prefatory' and 'Dec Supp' are special cases set to 1 January of the following year given
    in the publication.
    """

    # get the publication type
    pub_type = row[PUB_TYPE]

    # get the values from the row
    year = row[YEAR]
    month = row[MONTH]
    day = row[DAY]

    # Prefatory and Dec Supp were printed in the January ...
    if month == "Prefatory" or month == "Dec Supp":
        day = 1
        month = 1
        year += 1
    else:
        # convert String to numeric representation
        month = MONTH_MAP[month]

    # if the day isn't set, make it the first of the month
    if day is np.NaN:
        day = 1
        # if its a magazine, its actually published at the start of the following month
        if 'magazine' in pub_type:
            month += 1
            # if we have too may months, increment the year.
            if month == 13:
                month = 1
                year += 1

    # create a string in the expected format, YYYY-MM-DD
    month_str = "{:02d}".format(month)
    day_str = "{:02d}".format(day)
    date_str = "{}-{}-{}".format(year, month_str, day_str)

    # return the numpy object
    return np.datetime64(date_str)


def date_to_string(row):
    val = row[PRINTED_DATE]
    ts = pd.to_datetime(str(val))
    return ts.strftime('%Y-%m-%d')


def clean_attribution_type(row):
    val = row[ATTR_TYPE]
    if row is not np.NaN:
        val = str(val).strip()
        if attr_type_regex.match(val):
            val = val.replace(" ", "")
        elif val == 'm.d.e' or val == 'f.d.e':
            val = val + '.'
        elif val == 'pn':
            val = 'p/n'
    return val


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


def pickle_as_single_data_frame(max_year=settings.MAX_YEAR):
    """ Load all the pickle files as a single data frame. We don't return data beyond the MAX_YEAR value. """
    files = glob.glob(settings.PICKLE_SRC + '*.pickle', recursive=True)
    results = []
    for file in files:
        results.append(pd.read_pickle(file))

    # merge the files
    df = pd.concat(results, sort=False)

    # return data within the max year value
    return df[df[YEAR] <= max_year]


# ---------- Helper methods

def source_files_info_as_df():
    """ Create a data frame with the details of the source files """
    data = []
    for file in glob.glob(settings.DATA_SRC + '**/*.xlsx', recursive=True):
        row = [file.split('/')[-1]]
        data.append(row)
        df = pd.read_excel(file, sheet_name=POEM_DATA_SHEET)
        row.append(df.shape[0])
        row.append(df.shape[1])
    return pd.DataFrame(data=data, columns=['Source Files', 'Rows', 'Columns'])


def preprocessed_files_info_as_df():
    """ Create a data frame with the details of the preprocessed files """
    data = []
    for file in glob.glob(settings.PICKLE_SRC + '*.pickle', recursive=True):
        row = [file.split('/')[-1]]
        data.append(row)
        df = pd.read_pickle(file)
        row.append(df.shape[0])
        row.append(df.shape[1])
    return pd.DataFrame(data=data, columns=['Preprocessed Files', 'Rows', 'Columns'])


def complete_dataset():
    """" Get the complete dataset as a Pandas data frame."""
    return pickle_as_single_data_frame()


def start_year(df):
    """ Get the earliest year in the data frame """
    return df[YEAR].min()


def end_year(df):
    """ Get the latest year in the data frame """
    return df[YEAR].max()


def copied_poems(df):
    """" Create a new data frame of poems that have been identified as printed elsewhere """
    return df[df[REF_NO].notnull()]


def publication_list(df):
    """ Provide a list of publication titles in alphabetical order"""
    pub_name_list = df[PUB_TITLE].unique()
    return np.sort(pub_name_list)


def publications_df(df, pubs):
    """ Create a subset data frame based on the title of the publications """
    return df[df[PUB_TITLE].isin(pubs)]


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
    pubs.sort()

    # create a matrix of publications with a value of zero
    matrix_df = pd.DataFrame(np.zeros(shape=(pubs.size, pubs.size)), columns=pubs, index=pubs)

    # group by ref number (i.e the same poem)
    grouped = df.groupby(REF_NO)

    # go through the groups and update matrix
    for group_name, df_group in grouped:
        # go through the publication on one axis
        for pub_x in df_group[PUB_TITLE].unique():
            # go through the grotup on the other axes
            for pub_y in df_group[PUB_TITLE].unique():
                # increment score if the publications don't match
                if pub_y != pub_x:
                    matrix_df.at[pub_x, pub_y] += 1

    return matrix_df


def attribute_types_total_df(df):
    """ Data frame with details about attribution types """

    # keep track of the dataset total
    dataset_total = len(df.index)

    # group by, count and order the attribution types
    attr_type_count = df.groupby(ATTR_TYPE)[F_LINE].count().sort_values(ascending=False).to_frame()

    # reset the index and rename the count column
    attr_type_count = attr_type_count.reset_index().rename(columns={F_LINE: 'occurrences'})

    # add in the % calculation
    attr_type_count['% of total'] = (attr_type_count['occurrences'] / dataset_total) * 100

    return attr_type_count


def attribution_types_df(df):
    """ Create a data frame with details about attribution types """

    # create an empty data frame to hold the occurrences of attribution types by year
    # columns are the attribute types, the index will be the years
    cols = df[ATTR_TYPE].unique()
    index = np.arange(df[YEAR].min(), df[YEAR].max() + 1)
    attr_types = pd.DataFrame(np.NaN, index=index, columns=cols)

    # group by year and attribution type (reset data frame index)
    results = df.groupby([YEAR, ATTR_TYPE])[F_LINE].count().reset_index()

    # just group by year
    results_group_by = results.groupby(YEAR)

    # iterate over the years and unpack the tuples to get the data and update data frame
    for name, group in results_group_by:
        for row in group.iterrows():
            year = row[1][YEAR]
            attr = row[1][ATTR_TYPE]
            no = row[1][F_LINE]
            attr_types.at[year, attr] = no

    return attr_types


def publication_overview(df):
    """ Return a data frame that represents an overview of the data in a publication """

    # date range for that publication
    min_year = start_year(df)
    max_year = end_year(df)
    year_range_index = np.arange(min_year, max_year + 1, 1)

    # columns we want
    columns = np.array(['Total poems', 'Originals', 'Originals as %', 'Copies', 'Copies as %'])

    # create a results data frame filled with NaN
    results = pd.DataFrame(np.zeros(shape=(year_range_index.size, columns.size)), columns=columns,
                           index=year_range_index)

    for year in year_range_index:
        # get the subset for the year
        year_df = df[df[YEAR] == year]

        # total number of poems
        total = year_df[F_LINE].count()
        results.at[year, columns[0]] = total

        # total number of identified copies
        copies = year_df[REF_NO].notnull().sum()
        results.at[year, columns[3]] = copies

        # originals (not identified as a copy)
        originals = total - copies
        results.at[year, columns[1]] = originals

        # originals as a percentage of total
        originals_percent = originals / total * 100
        results.at[year, columns[2]] = originals_percent

        # copies as percent of total
        copies_percent = copies / total * 100
        results.at[year, columns[4]] = copies_percent

    return results

# ----------- Display widgets




def attributes_total_output(df, pub_title, out):
    pub_df = publications_df(df, [pub_title])
    attr_type_count_df = attribute_types_total_df(pub_df)

    # display results in a table
    out.clear_output()
    with out:
        display(HTML('<h3>{}, {}–{}</h3>'.format(pub_title, start_year(pub_df), end_year(pub_df))))
        display(HTML('<p>Attribute types in {}'.format(pub_title)))
        display(HTML(attr_type_count_df.to_html()))

        # display % in a plot
        attr_type_count_df.plot(kind='bar', x=ATTR_TYPE, y='% of total')
        plot.show()

        attr_types = attribution_types_df(pub_df)
        plot_attribution_types_line_plot(attr_types, "All attribution types in {} by year".format(pub_title))

        attr_types_subset = attr_types.drop(['nan', 'same', 'pseud', 'same (p/n)', '?', '--', 'f.pseud/f.d.e.'], axis=1,
                                            errors='ignore')
        plot_attribution_types_line_plot(attr_types_subset,
                                         "Attribution types against the whole dataset by year (non attributed and "
                                         "other artifacts removed)")

        ## TODO ... these might not exist ....!!!
        attr_types_subset['male'] = np.NaN
        if 'm.pseud' in attr_types_subset and 'm.d.e.' in attr_types_subset:
            attr_types_subset['male'] = attr_types_subset['m.pseud'] + attr_types_subset['m.d.e.']
        elif 'm.pseud' in attr_types_subset:
            attr_types_subset['male'] = attr_types_subset['m.pseud']
        elif 'm.d.e.' in attr_types_subset:
            attr_types_subset['male'] = attr_types_subset['m.d.e']

        attr_types_subset['female'] = np.NaN
        if 'f.pseud' in attr_types_subset and 'f.d.e.' in attr_types_subset:
            attr_types_subset['female'] = attr_types_subset['f.pseud'] + attr_types_subset['f.d.e.']
        elif 'f.pseud' in attr_types_subset:
            attr_types_subset['female'] = attr_types_subset['f.pseud']
        elif 'f.d.e.' in attr_types_subset:
            attr_types_subset['female'] = attr_types_subset['f.d.e.']

        attr_types_subset = attr_types_subset.drop(['m.pseud', 'm.d.e.', 'p/n', 'ini', 'f.d.e.', 'f.pseud'],
                                                   axis=1, errors='ignore')

        # regenerate graph
        plot_attribution_types_line_plot(attr_types_subset, "Gender attribution types by year")


def plot_attribution_types_line_plot(df, title, x_label='Years', y_label='Occurrences'):
    """ Plot the attribution types in a line plot """
    plot.figure(figsize=(20, 10))
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.title(title)
    sn.lineplot(data=df, dashes=False)
    plot.show()
