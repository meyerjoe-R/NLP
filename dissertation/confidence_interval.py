import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import re
import pandas as pd
import ast


def r_confidence_interval(r, n = 300):

    """
  r: corr
  n: size
  """
    #fisher's r to z
    r_z = np.arctanh(r)

    #se
    se = 1 / np.sqrt(n - 3)

    alpha = 0.05
    z = stats.norm.ppf(1 - alpha / 2)

    lo_z, hi_z = r_z - z * se, r_z + z * se

    #transform back
    lo, hi = np.tanh((lo_z, hi_z))

    return f"[{round(lo,2)}, {round(hi,2)}]"


def extract_pearson_r(text):
    pattern = r'statistic=([-+]?\d*\.\d+|\d+)'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    else:
        return text


def average_construct_performance(df):

    average_construct_performance = pd.concat(
        dfs.values()).groupby('construct')[['r', 'Lower Limit', 'Upper Limit'
                                            ]].mean().reset_index()

    return average_construct_performance


def create_apa_formatted_bar_chart_construct_average(averages,
                                                     conf_intervals,
                                                     labels,
                                                     title,
                                                     x_axis_label,
                                                     y_axis_label,
                                                     figure_number,
                                                     caption=None,
                                                     notes=None):
    """
    example:
    constructs = ['A_Scale_score', 'C_Scale_score', 'E_Scale_score', 'N_Scale_score', 'O_Scale_score']
    averages = [0.173175, 0.130078, 0.150775, 0.163335, 0.111570]
    conf_intervals = [(6.000000e-02, 0.2800), (1.750000e-02, 0.2400), (3.750000e-02, 0.2600), (5.500000e-02, 0.2700), (8.673617e-19, 0.2225)]
    reordered_constructs = ['E_Scale_score', 'A_Scale_score', 'O_Scale_score', 'C_Scale_score', 'N_Scale_score']
    reordered_averages = [averages[constructs.index(con)] for con in reordered_constructs]
    reordered_conf_intervals = [conf_intervals[constructs.index(con)] for con in reordered_constructs]

    create_apa_formatted_bar_chart_construct_average(reordered_averages, reordered_conf_intervals, reordered_constructs,
                                                    "Average Test Scores by Construct", "Construct", "Average Correlation", 1)

    """
    # Extract lower and upper bounds of confidence intervals
    lower_bounds, upper_bounds = zip(*conf_intervals)

    # Calculate the error (half the interval width)
    errors = [(upper - lower) / 2 for lower, upper in conf_intervals]

    # Create a bar chart with error bars
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x,
           averages,
           yerr=errors,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)

    # Add a note if provided
    if notes:
        plt.figtext(0.1, -0.1, notes, fontsize=10, ha='left')

    # Show the plot or save it to a file
    plt.tight_layout()

    # Add APA-style figure number and caption
    if caption:
        apa_caption = f"Figure {figure_number}. {caption}"
        plt.figtext(0.5, -0.15, apa_caption, fontsize=12, ha='center')

    # Show the plot or save it to a file
    plt.show()


def average_confidence_intervals(confidence_intervals):
    """
    example:
    intervals = [[0.1, 0.32], [0.07, 0.29], [-0.12, 0.1], [-0.02, 0.2], [0.01, 0.23]]
    average_confidence_intervals(intervals)
    
    """

    num_intervals = len(confidence_intervals)
    lower_sum = 0
    upper_sum = 0

    for interval in confidence_intervals:
        lower_sum += interval[0]
        upper_sum += interval[1]

    average_lower = round(lower_sum / num_intervals, 2)
    average_upper = round(upper_sum / num_intervals, 2)

    return [average_lower, average_upper]


def create_apa_formatted_bar_chart_model_average(averages,
                                                 conf_intervals,
                                                 labels,
                                                 title,
                                                 x_axis_label,
                                                 y_axis_label,
                                                 figure_number,
                                                 caption=None,
                                                 notes=None):
    """
    example:
    averages = [.18, 0.14, .03, 0.23]
    conf_intervals = [(0.01, 0.23), (0.03, 0.25), (-0.08, 0.14), (0.12, 0.33)]
    labels = ['BoW', 'Empath', 'LSTM', 'Big Bird']
    title = "Average Test Scores by Model"
    x_axis_label = "Model"
    y_axis_label = "Average Correlation"
    figure_number = 1
    caption = "Average test performance across constructs by model."

    # Call the function to create the APA-formatted bar chart
    create_apa_formatted_bar_chart(averages, conf_intervals, labels, title, x_axis_label, y_axis_label, figure_number)

    """

    # Extract lower and upper bounds of confidence intervals
    lower_bounds, upper_bounds = zip(*conf_intervals)

    # Calculate the error (half the interval width)
    errors = [(upper - lower) / 2 for lower, upper in conf_intervals]

    # Create a bar chart with error bars
    fig, ax = plt.subplots()
    x = np.arange(len(labels))
    ax.bar(x,
           averages,
           yerr=errors,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(y_axis_label)
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)

    # Add a note if provided
    if notes:
        plt.figtext(0.1, -0.1, notes, fontsize=10, ha='left')

    # Show the plot or save it to a file
    plt.tight_layout()

    # Add APA-style figure number and caption
    if caption:
        apa_caption = f"Figure {figure_number}. {caption}"
        plt.figtext(0.5, -0.15, apa_caption, fontsize=12, ha='center')

    # Show the plot or save it to a file
    plt.show()
