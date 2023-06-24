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
    df = (
    df.pipe(drop_loan_id_column)
      .pipe(fill_missing_values, ["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term", "Credit_History"], "LoanAmount")
      .pipe(one_hot_encode)
      .pipe(remove_outliers_z_score, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
      .pipe(remove_outliers_iqr, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"])
      .pipe(remove_skewness, ["ApplicantIncome", "CoapplicantIncome"])
      .pipe(scale_normalization)
      .pipe(oversample_minority_class)
)

    return df