import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pie_chart(df, i):
    """
    Plots a pie chart based on a specific column of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        i (str): The column name to use for the pie chart.

    Returns:
        None
    """
    sns.set_style('darkgrid')
    sns.set_palette('Set2')
    pd.pivot_table(df, values="Loan_ID", index=i, aggfunc="count").plot.pie(subplots=True)
    plt.show()

def plot_stacked_bar_chart(df, i):
    """
    Plots a stacked bar chart based on a specific column and Loan_Status of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        i (str): The column name to use for the stacked bar chart.

    Returns:
        None
    """
    sns.set_style('darkgrid')
    sns.set_palette('Set2')
    pd.pivot_table(df, values="Loan_ID", index="Loan_Status", columns=i, aggfunc="count").plot.bar(subplots=False, stacked=True)
    plt.show()

def plot_scatter_plot(df, x, y):
    """
    Plots a scatter plot based on two columns of a DataFrame and differentites the data with colour based on the Loan_Status column.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.

    Returns:
        None
    """
    colours = {"Y": "green", "N": "red"}
    plt.scatter(x=df[x], y=df[y], c=df["Loan_Status"].apply(lambda x: colours[x]))
    plt.show()

def plot_histogram(df, x):
    """
    Plots a histogram based on a specific column of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x (str): The column name to use for the histogram.

    Returns:
        None
    """
    sns.histplot(df, x=x, kde=True)
    plt.show()

def plot_violin_plot(df, x, y):
    """
    Plots a violin plot based on two columns of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.

    Returns:
        None
    """
    sns.violinplot(df, x=x, y=y)
    plt.show()

def plot_box_plot(df, x, y):
    """
    Plots a box plot based on two columns of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        x (str): The column name for the x-axis.
        y (str): The column name for the y-axis.

    Returns:
        None
    """
    sns.boxplot(df, x=x, y=y)
    plt.show()