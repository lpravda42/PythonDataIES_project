from steps.data_processing_funcs import *

def data_processing_function(df):
    """Performs a series of data processing steps on a DataFrame.

    This function applies a series of data processing steps to the provided DataFrame. The steps include dropping the 'Loan_ID' column, filling missing values in specific columns using the
    mode and mean, one-hot encoding categorical columns, removing outliers using z-scores and IQR, removing skewness
    from numerical columns, performing scale normalization, and oversampling the minority class.

    Args:
        df (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    #calculate_missing_percentage(df)
    df = drop_loan_id_column(df)
    fill_missing_values(df, ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"], "LoanAmount")
    df = one_hot_encode(df)
    df = remove_outliers_z_score(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
    df = remove_outliers_iqr(df, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
    df = remove_skewness(df, ["ApplicantIncome", "CoapplicantIncome"])
    df = scale_normalization(df)
    df = oversample_minority_class(df)

    return df