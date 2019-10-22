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

