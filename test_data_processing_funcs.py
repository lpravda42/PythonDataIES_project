import pandas as pd
import pytest
from data_processing_funcs import *

def test_calculate_missing_percentage(capsys):
    df = pd.DataFrame({"A": [None, None, None, None],
                       "B": [None, None, None, 8.0],
                       "C": [9.0, 10.0, 11.0, None]})

    calculate_missing_percentage(df)
    captured = capsys.readouterr()
    expected_output = "Percentage of missing values in the column A: 100.0\n" \
                      "Percentage of missing values in the column B: 75.0\n" \
                      "Percentage of missing values in the column C: 25.0\n"

    assert captured.out == expected_output
    assert captured.err == ""
    
def test_fill_missing_values():
    df = pd.DataFrame({"A": ["Yes", "No", "No", None],
                       "B": [None, "7", "8", "8"],
                       "C": [9.0, 10.0, 11.0, None]})
    
    fill_missing_values(df, ["A", "B"], "C")
    expected_df = pd.DataFrame({"A": ["Yes", "No", "No", "No"],
                                "B": ["8", "7", "8", "8"],
                                "C": [9.0, 10.0, 11.0, 10.0]})
    
    pd.testing.assert_frame_equal(df, expected_df)

def test_remove_outliers_z_score():
    df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000000000000000.0, 1.0],
                       "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000000000000000.0, 1.0]})
    
    df = remove_outliers_z_score(df, ["A"]).reset_index(drop = True)
    expected_df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).reset_index(drop = True)
    
    pd.testing.assert_frame_equal(df, expected_df)

def test_remove_outliers_iqr():
    df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000000000000000.0, 1.0],
                       "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10000000000000000.0, 1.0]})
    
    df = remove_outliers_iqr(df, ["A"]).reset_index(drop = True)
    expected_df = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                                "B": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}).reset_index(drop = True)
    
    pd.testing.assert_frame_equal(df, expected_df)