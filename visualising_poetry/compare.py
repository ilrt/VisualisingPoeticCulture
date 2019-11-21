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


def compare():

    # print the start date/time of the job
    print(datetime.datetime.utcnow())

    with open('results.csv', 'a') as csv_file:

        # create a CSV writer
        results_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write the columns for the results
        results_writer.writerow(['Publication (a)', 'Index (a)', 'First line (a)', 'Ref no. (a)',
                                 'Publication (b)', 'Index (b)', 'First line (b)', 'Ref no. (b)', 'Ratio'])

        # get the dataset
        df = vpd.complete_dataset()

        # reset the index
        df = df.reset_index(drop=True)

        # find the column index for items we are interested in
        col_f_line_idx = df.columns.get_loc(vpd.F_LINE)
        col_pub_idx = df.columns.get_loc(vpd.PUB_TITLE)
        col_ref_idx = df.columns.get_loc(vpd.REF_NO)

        # start and end parameters
        start = 0
        end = len(df.index)
        index_range = np.arange(start, end)

        # go through the index range ...
        for x in index_range:
            # get first line, publication and ref number of poem a
            string_a = df.iat[x, col_f_line_idx]
            pub_a = df.iat[x, col_pub_idx]
            ref_no = df.iat[x, col_ref_idx]
            # inner loop: start 1 on from the outer (avoids duplication)
            for y in np.arange(start + 1, end):
                # don't check the same index
                if x != y:
                    # get the first line of poem b and compare with a
                    string_b = df.iat[y, col_f_line_idx]
                    ratio = fuzz.ratio(string_a, string_b)
                    #
                    if ratio > RATIO_THRESHOLD:
                        pub_b = df.iat[y, col_pub_idx]
                        ref_no_b = df.iat[x, col_ref_idx]
                        results_writer.writerow([pub_a, x, string_a, ref_no, pub_b, y, string_b, ref_no_b, ratio])

    # print the start end/time of the job
    print(datetime.datetime.utcnow())
