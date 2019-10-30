import visualising_poetry.data as vpd

import numpy as np
import matplotlib.pyplot as plot
import ipywidgets as widgets
from IPython.display import display, HTML


def publication_list_widget(df):
    """ Create a widget with a list of available publications """
    pub_list = vpd.publication_list(df)
    return widgets.Select(
        options=pub_list,
        description="Choose",
        disabled=False
    )


def publication_year_widget(df):
    """ Create a widget with a list of available years """
    start_year = vpd.start_year(df)
    end_year = vpd.end_year(df)
    year_range = np.arange(start_year, end_year + 1)
    return widgets.Select(
        options=year_range,
        description="Choose",
        disabled=False
    )


def publication_overview_selection(df, value, out):

    # get the publications data
    pub_df = vpd.publications_df(df, [value])
    results = vpd.publication_overview(pub_df)

    with out:
        # date range for the publication
        start_date = vpd.start_year(pub_df)
        end_date = vpd.end_year(pub_df)
        years = np.arange(start_date, end_date + 1)
        title = '<h3>Poems and copes in {}, {}–{}</h3>'.format(value, start_date, end_date)
        display(HTML(title))
        display(HTML(results.to_html()))
        plot.figure(figsize=(10, 5))
        plot.plot(results[results.columns[2]])
        plot.plot(results[results.columns[4]])
        plot.legend([results.columns[2], results.columns[4]])
        plot.xticks(years, years)
        plot.xlabel("Years")
        plot.ylabel("% of total poems")
        plot.show()


def author_list_widget(df):
    author_list = vpd.unique_author_list(df)

    return widgets.Select(
        options=author_list,
        description="Choose",
        disabled=False
    )
