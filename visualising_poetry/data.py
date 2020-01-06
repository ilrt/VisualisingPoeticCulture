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
S_LINE = 'second line'
P_LINE = 'penultimate line'
L_LINE = 'last line'
ATTR_TYPE = 'attribution type'
AUTH = 'authorship'

# computed
PRINTED_DATE = 'printed'
PRINTED_DATE_STR = 'printed (string)'
PRINTED_YEAR = 'printed year'
GENDER = 'gender'

# used to convert strings used in the dataset to numbers
MONTH_MAP = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12, 'Dec Supp': 12, 'Prefatory': 1
}

# expected attribution types
EXPECTED_ATTRS = ['nan', 'p/n', 'ini', 'm.pseud', 'm.d.e.', 'f.pseud', 'f.d.e.', 'information missing', 'illegible']

# summary data - useful subset of fields
SUMMARY = [REF_NO, PUB_TITLE, YEAR, MONTH, DAY, PRINTED_DATE_STR, AUTH, F_LINE, ATTR_TYPE]

# used to tidy up m.pseud and f.pseud
attr_type_regex = re.compile(r'^[m|f]\. pseud$')

# used to check if first names are just initials
initials_regex = re.compile(r'^((\b[A-Za-z]\b){1}(\.)?( )?)+')

# used to check if a name has a male prefix
male_prefix_regex = re.compile(r'^\b(Mr|Dr|Rev)\b(\.)?')

# used to check if we have a female prefix
female_prefix = re.compile(r'^\b(Mrs|Miss)\b(\.)?')

# used to check if we have one of the common female names
female_names = \
    re.compile(
        r'^\b(Amy|Anna|Anne|Aphra|Catherine|Charlotte|Elizabeth|Esther|Jenny|Judith|Mary|Martha|Mary|Sarah|Frances)\b')

male_identifiers = ['Bishop', 'Lord', 'Duke', 'Signor']

# computed values when working out gender
GENDER_MALE = 'male'
GENDER_FEMALE = 'female'
GENDER_UNKNOWN = 'not attributed'
GENDER_AMBIGUOUS = 'attributed (ambiguous)'


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


def clean_reports():
    """ Clean the reports folder"""
    clean(settings.REPORTS_DIR)


def clean_all():
    """ Clean the environment """
    clean_preprocessed()
    clean_source()
    clean_reports()


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
        print(file)
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

        # add the print year for convenience
        df[PRINTED_YEAR] = df[PRINTED_DATE].dt.year

        # clean the attribution type
        df[ATTR_TYPE] = df.apply(clean_attribution_type, axis=1)

        # calculate gender
        df[GENDER] = df.apply(calculate_gender, axis=1)

        # strip extra whitespace in authors
        df[AUTH] = df.apply(clean_author, axis=1)

        df[F_LINE] = df[F_LINE].str.strip()

        # write the pickle file
        df.to_pickle(settings.PICKLE_SRC + filename + '.pickle')


def create_report_dir():
    """ Create a report directory """
    if not os.path.exists(settings.REPORTS_DIR):
        os.makedirs(settings.REPORTS_DIR)


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


def clean_author(row):
    val = row[AUTH]
    if row is not np.NaN:
        val = str(val).strip()
    return val


def calculate_gender(row):
    """ An attempt to use the 'attribution type' and 'authorship' values to classify the gender of the author."""

    # start with the attribution
    attr = row[ATTR_TYPE]
    # nan means uncredited, so we don't know
    if attr == 'nan':
        return GENDER_UNKNOWN
    # clearly identified as male
    elif attr == 'm.d.e.' or attr == 'm.pseud':
        return GENDER_MALE
    # clearly identified as female
    elif attr == 'f.d.e.' or attr == 'f.pseud':
        return GENDER_FEMALE
    # otherwise ... we need to make an education guess based on the
    else:
        name = row[AUTH]
        # an attribution type is given, but no authorship, so its ambiguous
        if not type(name) is str:
            return GENDER_AMBIGUOUS
        else:
            # names are generally given as [LAST NAME], [FIRST NAME]
            # first name and title gives us the best chance
            tmp = name.split(', ')
            # not too parts, so its ambiguous
            if len(tmp) < 2:
                return GENDER_AMBIGUOUS
            else:
                f_name = tmp[1].strip()
                # just initials? ambiguous ...
                if re.match(initials_regex, f_name):
                    return GENDER_AMBIGUOUS
                # male prefixes?
                elif re.match(male_prefix_regex, f_name):
                    return GENDER_MALE
                # female prefixes?
                elif re.match(female_prefix, f_name):
                    return GENDER_FEMALE
                # common female names?
                elif re.match(female_names, f_name):
                    return GENDER_FEMALE
                # reference to a maiden name?
                elif 'nee ' in f_name:
                    return GENDER_FEMALE
                # balance of probability, what's left is male
                else:
                    return GENDER_MALE


# ---------- Methods for setting an environment up for a notebook

def setup():
    """ Setup the environment so the notebooks have data. """
    # clean directories
    clean_source()
    clean_preprocessed()
    clean_reports()
    # download the data
    download_data()
    unpack_zip()
    # create panda data frames and pickle them
    write_pickle_data_frames()
    # create reports directory
    create_report_dir()


def setup_if_needed():
    """ Helper method to run at the start of a notebook to ensure that there is data """
    if not os.path.exists(settings.DATA_SRC) or not os.path.exists(settings.DATA_SRC + settings.DB_FILE):
        print("No zip file. Getting data.")
        setup()
        print("Setup complete")
    # create reports directory
    create_report_dir()


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


def year_range(df):
    """ Create an array of print years in a data frame (used in graphs a lot) """
    return np.arange(start_year(df), end_year(df) + 1)


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


def authors_df(df, authors):
    """ Create a subset data frame based on the authors of the publications """
    _authors_df = df[df[AUTH].isin(authors)]
    _authors_df = _authors_df.sort_values([PRINTED_DATE_STR])
    _authors_df = _authors_df.reset_index(drop=True)
    return _authors_df


def authors_expanded(df, authors):
    """ We 'expand' the dataset for authors by including copies of poems where they aren't
        directly attributed to the author, but other copies with the same reference number
        that have been attributed to that author. """

    _authors_df = authors_df(df, authors)
    ids = _authors_df[_authors_df[REF_NO].notnull()][REF_NO].unique()
    expanded = pd.concat([_authors_df, (df[df[REF_NO].isin(ids) & ~df[AUTH].isin(authors)])])
    expanded = expanded.sort_values([PRINTED_DATE_STR])
    expanded = expanded.reset_index(drop=True)
    return expanded


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


def unexpected_attribute_types(df):
    """ Data checking method – give poem details if they have unexpected attribute types """
    return df[~df[ATTR_TYPE].isin(EXPECTED_ATTRS)][SUMMARY]


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


def attribution_types_overview_df(df):
    """ Create a data frame with details with an overview of attribution types by year. Thus will include
        the raw numbers and % of total for that year. """

    # We want two columns for each attribute type, with a raw number and %. We need to create the passed in
    # data frame to discover what attributes types are available do we can create the columns for the
    # new data frame

    # format strings for columns
    no_str = "{} as no."
    pc_str = "{} as %"

    # hold total and attribute types
    all_cols = ['Total']

    # create columns based on attributes
    attrs = df[ATTR_TYPE].unique()
    for attr in attrs:
        all_cols.append(no_str.format(attr))
        all_cols.append(pc_str.format(attr))

    # create an empty data frame
    columns = np.array(all_cols)
    index = year_range(df)
    attr_types_df = pd.DataFrame(np.NaN, index=index, columns=columns)

    # group dataset by year
    results_group_by = df.groupby([PRINTED_YEAR])

    # process each year
    for year, year_group in results_group_by:
        # record the total for the year
        attr_group_year_total = year_group[PRINTED_YEAR].count()
        attr_types_df.at[year, columns[0]] = attr_group_year_total
        # group by attribute type for the year
        attr_group_year = year_group.groupby(ATTR_TYPE)
        # go through each attribute type ...
        for attr_name, attr_group in attr_group_year:
            attr_no_key = no_str.format(attr_name)
            attr_pc_key = pc_str.format(attr_name)
            attr_group_total = attr_group[ATTR_TYPE].count()
            attr_group_pc = (attr_group_total / attr_group_year_total) * 100
            attr_types_df.at[year, attr_no_key] = attr_group_total
            attr_types_df.at[year, attr_pc_key] = attr_group_pc

    return attr_types_df


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


def gender_overview_df(df):
    """ Create a data frame providing an overview of poems and gender for a provided data frame """

    # printed date slips into Jan beyond MAX year, let's remove those
    df = df[df[PRINTED_YEAR] <= settings.MAX_YEAR]

    # date range for that publication
    min_year = start_year(df)
    max_year = end_year(df)
    year_range_index = np.arange(min_year, max_year + 1, 1)

    # columns we want
    columns = np.array(['Total poems', 'Male', 'Male as %', 'Female', 'Female as %', 'N/A', 'N/A as %',
                        'Ambiguous', 'Ambiguous as %'])

    # create a matrix of titles and possible years with a value of zero
    gender_df = pd.DataFrame(np.zeros(shape=(year_range_index.size, columns.size)), columns=columns,
                             index=year_range_index)

    # sort and group the data
    results = df.sort_values([PRINTED_YEAR, GENDER])[[PRINTED_YEAR, GENDER]]
    results_group_by = results.groupby(PRINTED_YEAR)

    for name, group in results_group_by:
        # total for that year
        total = group[PRINTED_YEAR].count()
        # count each of the four types
        male = group[group[GENDER] == GENDER_MALE][GENDER].count()
        female = group[group[GENDER] == GENDER_FEMALE][GENDER].count()
        na = group[group[GENDER] == GENDER_UNKNOWN][GENDER].count()
        amb = group[group[GENDER] == GENDER_AMBIGUOUS][GENDER].count()
        # add raw data and % to the results
        gender_df.at[name, columns[0]] = total
        gender_df.at[name, columns[1]] = male
        gender_df.at[name, columns[2]] = (male / total) * 100
        gender_df.at[name, columns[3]] = female
        gender_df.at[name, columns[4]] = (female / total) * 100
        gender_df.at[name, columns[5]] = na
        gender_df.at[name, columns[6]] = (na / total) * 100
        gender_df.at[name, columns[7]] = amb
        gender_df.at[name, columns[8]] = (amb / total) * 100

    return gender_df


def authorship_overview_df(df):
    """ Create a data frame of author and number of poems attributed to them """
    authors = df.groupby(AUTH)[F_LINE].count().reset_index()
    authors = authors.sort_values(F_LINE, ascending=False)
    authors.columns = ['Author', 'Total Poems']
    authors = authors.reset_index(drop=True)
    authors = authors.drop([0])
    print(type(authors))
    return authors


def authorship_poems_overview_df(df):
    """ Provide a data frame that shows the no. authors and the total number of poems attributed to them, for
        example 20 poets are attributed to 15 poems each. """

    # get a running total for each other
    authors = authorship_overview_df(df)

    # group by the poem totals
    authors_group = authors.groupby(authors.columns[1])

    # hold the data
    data = []

    # columns in the new data frame
    columns = ['No. of Poems', 'No. of Authors']

    # iterate each other and count poems, add to the new data set
    for name, group in authors_group:
        row = [name, group['Total Poems'].count()]
        data.append(row)

    # return a pandas data frame
    return pd.DataFrame(data, columns=columns)


def authorship_publications_overview_df(df, pub_no=20):
    """ Provide a data frame that shows which authors appeared in which journals. The pub_no, is a threshold
        to only show authors attributed to that or more poems in the dataset. Default is 20 or more. """

    # get authors
    authors = authorship_overview_df(df)

    # limit it to authors that meet the the threshold
    authors = authors[authors['Total Poems'] >= 20]

    # set the index to the authors
    authors = authors.set_index('Author')

    # get a list of publications
    pubs = df[PUB_TITLE].unique().tolist()

    # create a data frame with columns based on the publications and the index based on the authors
    author_pubs = pd.DataFrame(np.zeros(shape=(len(authors.index), len(pubs))), columns=pubs, index=authors.index)

    for author in author_pubs.index:
        author_pub_df = df[df[AUTH] == author]
        for pub in author_pub_df[PUB_TITLE]:
            author_pubs.at[author, pub] += 1

    return author_pubs


def authorship_publication_year_overview_df(df, pub_no=20):
    """ Provide a data frame that shows which authors poems were printed in each year. The pub_no, is a threshold
        to only show authors attributed to that or more poems in the dataset. Default is 20 or more. """

    # get authors
    authors = authorship_overview_df(df)

    # limit it to authors with 20 or more poems
    authors = authors[authors['Total Poems'] >= pub_no]

    # year range
    years = year_range(df)

    # set the index to the authors
    authors = authors.set_index('Author')

    # data frame for authors and years
    authors_years = pd.DataFrame(np.zeros(shape=(len(authors.index), len(years))), columns=years, index=authors.index)

    # populate the data frame
    for author in authors_years.index:
        author_year_df = df[df[AUTH] == author]
        for year in author_year_df[PRINTED_YEAR]:
            if year in authors_years.columns:
                authors_years.at[author, year] += 1

    return authors_years


def unique_author_list(df, pub_no=2):
    """ Get a list of authors who have published a specified number of poems or greater.
        By default, 2 or more poems. We filter our nan."""

    # get authors
    authors = authorship_overview_df(df)
    # filter the data
    authors = authors[authors['Total Poems'] >= pub_no]
    authors = authors[authors['Author'] != 'nan']

    # get authors and filter
    author_list = authors['Author'].to_list()
    author_list.sort()

    return author_list


def author_publications(df, author):
    """ Create a data frame of poems for the specified author """

    # get the data for this author
    author_df = authors_expanded(df, [author])

    # group the data by publications
    expanded_pub_group = author_df.groupby(PUB_TITLE)

    # get the year range
    year_range_author = np.arange(author_df[PRINTED_YEAR].min(), author_df[PRINTED_YEAR].max() + 1)

    # publications the author is associated with
    publications_author = author_df[PUB_TITLE].unique()
    publications_author.sort()

    # create a data frame populated with zeros
    matrix = pd.DataFrame(np.zeros(shape=(len(publications_author), len(year_range_author))), columns=year_range_author,
                          index=publications_author)

    # update the matrix
    for name, group in expanded_pub_group:
        years = group[PRINTED_YEAR]
        for year in years.to_list():
            matrix.at[name, year] += 1

    return matrix


def author_unique_vs_copies(df, author):
    """ Get statistics on unique vs copies for a particular author """

    author_df = authors_expanded(df, [author])

    total_appearances = len(author_df.index)

    appearing_once = author_df[REF_NO].isna().sum()
    more_than_once = author_df[REF_NO].nunique()
    total_poems = more_than_once + appearing_once
    copies = total_appearances - appearing_once

    return pd.DataFrame({'Total appearances': [total_appearances], 'No. of poems': [total_poems],
                         'No. of poems appearing once': [appearing_once],
                         'No. of poems with copies': [more_than_once], 'No. of copies': [copies]},
                        index=[author])


def author_poem_matrix(df, author):
    """ Create a matrix to show which poems were published on which"""

    author_df = authors_expanded(df, [author])

    # hold our interim results
    interim_results = {}

    # first, we deal with those identified as copies. The first line can change
    # slightly, but we will just use one of the group as the key/index
    author_df_group = author_df.groupby(REF_NO)

    # group through the copies
    for name, group in author_df_group:
        # track title and years
        poem_title = None
        years = []

        for index, row in group.iterrows():
            if poem_title is None:
                poem_title = row[F_LINE]
            years.append(row[PRINTED_YEAR])

        # add to the results
        interim_results[poem_title] = years

    # now we need to deal with the results
    uniques = author_df[author_df[REF_NO].isnull()][[F_LINE, PRINTED_YEAR]]

    for index, row in uniques.iterrows():
        interim_results[row[F_LINE]] = [row[PRINTED_YEAR]]

    min_year = author_df[PRINTED_YEAR].min()
    max_year = author_df[PRINTED_YEAR].max()
    year_range_index = np.arange(min_year, max_year + 1)

    matrix = pd.DataFrame(np.zeros(shape=(len(interim_results.keys()), year_range_index.size)),
                          columns=year_range_index, index=interim_results.keys())

    for poem in interim_results.keys():
        for year in interim_results[poem]:
            matrix.at[poem, year] += 1

    matrix = matrix.sort_index(ascending=True)

    return matrix

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
