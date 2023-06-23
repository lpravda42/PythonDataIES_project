import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
import os
import joblib

def calculate_missing_percentage(df):
    """Calculates the percentage of missing values in each column of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to analyze.

    Returns:
        None
    """
    for i in df.columns:
        print(f"Percentage of missing values in the column {i}: {df[i].isnull().sum() * 100 / len(df.index)}")


def drop_loan_id_column(df):
    """Drops the 'Loan_ID' column from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.

    Returns:
        pandas.DataFrame: The modified DataFrame without the 'Loan_ID' column.
    """
    return df.drop(["Loan_ID"], axis = 1)


def fill_missing_values(df, str_columns, num_column):
    """Fills missing values in specified columns of a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): List of column names to fill missing values.

    Returns:
        None
    """
    for i in str_columns:
        df[i].fillna(df[i].mode()[0], inplace = True)

    df[num_column].fillna(df[num_column].mean(), inplace = True)


def one_hot_encode(df):
    """Performs one-hot encoding on a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to encode.

    Returns:
        pandas.DataFrame: The encoded DataFrame.
    """
    df = pd.get_dummies(df)
    df = df.drop(["Gender_Female", "Married_No", "Loan_Status_N"], axis = 1)
    return df


def remove_outliers_z_score(df, columns):
    """Removes outliers from specified columns using z-score method.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): List of column names to remove outliers.

    Returns:
        pandas.DataFrame: The modified DataFrame without outliers.
    """
    for i in columns:
        z_scores = np.abs(stats.zscore(df[i]))
        df = df[(z_scores < 3)]

    return df


def remove_outliers_iqr(df, columns):
    """Removes outliers from specified columns using IQR method.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): List of column names to remove outliers.

    Returns:
        pandas.DataFrame: The modified DataFrame without outliers.
    """
    for i in columns:
        q1 = df[i].quantile(0.25)
        q3 = df[i].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[i] >= lower_bound) & (df[i] <= upper_bound)]

    return df


def remove_skewness(df, columns):
    """Applies square root transformation to specified columns to reduce skewness.

    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        columns (list): List of column names to apply square root transformation.

    Returns:
        pandas.DataFrame: The modified DataFrame with reduced skewness.
    """
    for i in columns:
        df[i] = np.sqrt(df[i])

    return df


def scale_normalization(df):
    """Normalizes the scale of the features of a DataFrame using Min-Max scaling.

    Args:
        df (pandas.DataFrame): The DataFrame to scale.

    Returns:
        pandas.DataFrame: The scaled DataFrame.
    """
    scaler = MinMaxScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    joblib.dump(scaler,'./steps/scaler.pkl')
    return scaled_df


def oversample_minority_class(df):
    """Performs oversampling of the minority class using RandomOverSampler.

    Args:
        df (pandas.DataFrame): The DataFrame to oversample.

    Returns:
        pandas.DataFrame: The oversampled DataFrame.
    """
    X = df.drop(["Loan_Status_Y"], axis = 1)
    y = df["Loan_Status_Y"]
    oversample = RandomOverSampler(sampling_strategy = "minority")
    oversample.fit_resample(X, y)
    df = pd.concat([X, y], axis = 1)

    return df