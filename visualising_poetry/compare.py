"""
    Visualising Poetry module for looking for copies of poems within the dataset. This examines the dataset
    in the pickle files rather than than the Excel files directly.
"""
from fuzzywuzzy import fuzz
import visualising_poetry.data as vpd
import numpy as np
import datetime
import csv

RATIO_THRESHOLD = 80

HEADER = ['Publication (a)', 'Year (a)', 'Month (a)', 'Day (a)', 'Index (a)', 'Ref no. (a)', 'First line (a)',
          'Second line (a)', 'Penultimate line (a)', 'Last line (a)', 'Publication (b)', 'Year (b)', 'Month (b)',
          'Day (b)', 'Index (b)', 'Ref no. (b)', 'First line (b)', 'Second line (b)', 'Penultimate line (b)',
          'Last line (b)', 'Ratio']

REPORT_FILE = 'reports/match_results.csv'


def compare(threshold=RATIO_THRESHOLD):
    """ Run the poem comparison job """

    # print the start date/time of the job
    print('Job started: ' + str(datetime.datetime.utcnow()))

    # get the dataset
    df = vpd.complete_dataset()

    # reset the index
    df = df.reset_index(drop=True)

    # find the column index for items we are interested in
    col_f_line_idx = df.columns.get_loc(vpd.F_LINE)
    col_pub_idx = df.columns.get_loc(vpd.PUB_TITLE)
    col_ref_idx = df.columns.get_loc(vpd.REF_NO)
    col_s_line_idx = df.columns.get_loc(vpd.S_LINE)
    col_p_line_idx = df.columns.get_loc(vpd.P_LINE)
    col_l_line_idx = df.columns.get_loc(vpd.L_LINE)
    col_year_idx = df.columns.get_loc(vpd.YEAR)
    col_mon_idx = df.columns.get_loc(vpd.MONTH)
    col_day_idx = df.columns.get_loc(vpd.DAY)

    with open(REPORT_FILE, 'w') as csv_file:

        # create a CSV writer
        results_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write the columns for the results
        results_writer.writerow(HEADER)

        # start and end parameters
        index_range = np.arange(0, len(df.index))

        # go through the index range ...
        for x in index_range:
            # get first line, publication and ref number of poem a
            string_a = df.iat[x, col_f_line_idx]

            # inner loop: start +1 on from the outer (avoids duplication)
            for y in np.arange(x + 1, len(df.index)):
                # don't check the same index
                # get the first line of poem b and compare with a
                string_b = df.iat[y, col_f_line_idx]
                ratio = fuzz.ratio(string_a, string_b)
                if ratio >= threshold:
                    # a values
                    pub_a = df.iat[x, col_pub_idx]
                    year_a = df.iat[x, col_year_idx]
                    month_a = df.iat[x, col_mon_idx]
                    day_a = df.iat[x, col_day_idx]
                    ref_no_a = df.iat[x, col_ref_idx]
                    s_line_a = df.iat[x, col_s_line_idx]
                    p_line_a = df.iat[x, col_p_line_idx]
                    l_line_a = df.iat[x, col_l_line_idx]
                    # b values
                    pub_b = df.iat[y, col_pub_idx]
                    year_b = df.iat[y, col_year_idx]
                    month_b = df.iat[y, col_mon_idx]
                    day_b = df.iat[y, col_day_idx]
                    ref_no_b = df.iat[y, col_ref_idx]
                    s_line_b = df.iat[y, col_s_line_idx]
                    p_line_b = df.iat[y, col_p_line_idx]
                    l_line_b = df.iat[y, col_l_line_idx]

                    results_writer.writerow([pub_a, year_a, month_a, day_a, x, ref_no_a, string_a, s_line_a,
                                             p_line_a, l_line_a, pub_b, year_b, month_b, day_b, y, ref_no_b, string_b,
                                             s_line_b, p_line_b, l_line_b, ratio])

    # print the start end/time of the job
    print('Job finished:' + str(datetime.datetime.utcnow()))
    print('Report written to ' + REPORT_FILE)
