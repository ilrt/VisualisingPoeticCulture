import settings as settings
import visualising_poetry.data as vpd

from datetime import datetime
import pandas as pd


def poems_unexpected_multiple_authors_report():
    """ Create a CSV file with details of poems that have more than one unique author. This is a useful
        report for increasing the attributed authors across the dataset, since an author might have
        been identified in one copy but not the other. It also helps improving inconsistencies in
        naming conventions. """

    # path and name for the report
    file_name = "poems_unexpected_multiple_authors–{}.csv".format(datetime.now().strftime('%Y-%m-%d'))
    full_path_name = settings.REPORTS_DIR + file_name

    # get the copies
    df = vpd.complete_dataset()
    df_copies = vpd.copied_poems(df)
    df_copies_group = df_copies.groupby(vpd.REF_NO)

    # use to track poems IDs we are interested in
    track_ids = []

    # track poems with more than one unique author
    for name, group in df_copies_group:
        if len(group[vpd.AUTH].unique()) > 1:
            track_ids.append(name)

    copies_with_multiple_authors = df[df[vpd.REF_NO].isin(track_ids)].sort_values(
        [vpd.REF_NO, vpd.PRINTED_DATE, vpd.AUTH])[vpd.SUMMARY]

    # write file
    copies_with_multiple_authors.to_csv(full_path_name)

    return "{} written to {}".format(file_name, settings.REPORTS_DIR)
