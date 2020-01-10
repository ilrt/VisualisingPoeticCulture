import visualising_poetry.data as vpd
import visualising_poetry.plot as vpp

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
        vpp.attribution_types_line_plot(attr_types, "All attribution types in {} by year".format(pub_title))

        attr_types_subset = attr_types.drop(['nan', 'same', 'pseud', 'same (p/n)', '?', '--', 'f.pseud/f.d.e.'], axis=1,
                                            errors='ignore')
        vpp.attribution_types_line_plot(attr_types_subset,
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
        vpp.attribution_types_line_plot(attr_types_subset, "Gender attribution types by year")
