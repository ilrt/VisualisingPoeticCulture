import numpy as np
import matplotlib.pyplot as plot
import seaborn as sn


def attribution_types_line_plot(df, title, x_label='Years', y_label='% of total'):
    """ Attribution types in a line plot, showing % of total for each year  """
    plot.figure(figsize=(20, 10))
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.title(title)
    lp = sn.lineplot(data=df, legend='full', palette='colorblind', dashes=False)
    lp.set(xticks=df.index.values)
    plot.show()


def gender_line_plot(df, title="Attribution type by gender", x_label='Years', y_label='% of total'):
    """ Plot the gender """
    plot.figure(figsize=(20, 10))
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.title(title)
    lp = sn.lineplot(data=df[['Male as %', 'Female as %', 'Ambiguous as %', 'N/A as %']],
                     legend='full', palette='colorblind', dashes=False)
    lp.set(xticks=df.index.values)
    plot.show()


def author_poems_plot(df, title="No. of authors / No. of poems", x_label="No. of Poems", y_label='No. of authors'):
    """ Show the number of authors against number of poems attributed to them in a scatter plot """

    sn.set()
    plot.figure(figsize=(20, 10))
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.title(title)
    sn.lineplot(y='No. of Authors', x='No. of Poems', data=df)
    plot.show()


def authorship_publications_overview_plot(df):
    """ Show authors and the number of times they were published in the various publications. """

    plot.figure(figsize=(20, 20))
    with sn.axes_style("white"):
        sn.heatmap(df, annot=True, cmap='Oranges', cbar=False, fmt='g', annot_kws={'size': 14})
    plot.show()


def authorship_publication_year_plot(df):
    """ Show authors and the number of times they were published in a year. """

    plot.figure(figsize=(18, 18))
    with sn.axes_style("white"):
        hm = sn.heatmap(df, annot=True, cmap='Oranges', cbar=False, fmt='g', annot_kws={'size': 14})
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
    plot.show()


def heat_map_plot(df, figsize=(18, 18), title='Heatmap', axes_style='white', cmap='Oranges'):
    """ General heat map plot """
    plot.figure(figsize=figsize)
    with sn.axes_style(axes_style):
        hm = sn.heatmap(df, annot=True, cmap=cmap, cbar=False, fmt='g', annot_kws={'size': 14})
        hm.set_yticklabels(hm.get_yticklabels(), rotation=0)
        hm.set_title(title)
    plot.show()
