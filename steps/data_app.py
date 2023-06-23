import pandas as pd
import joblib
import os

from steps.data_processing_funcs import scale_normalization 

def process_data_app(raw_data):
    """_summary_

    Args:
        raw_data (_type_): _description_
    """
    raw_data_df = pd.DataFrame(raw_data, index=[0])
    
    desired_columns = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Gender_Male', 'Married_Yes', 'Dependents_0', 'Dependents_1', 'Dependents_2', 'Dependents_3+',
        'Education_Graduate', 'Education_Not Graduate', 'Self_Employed_No', 'Self_Employed_Yes',
        'Property_Area_Rural', 'Property_Area_Semiurban', 'Property_Area_Urban'
    ]
    
    # Create a new DataFrame with the desired columns
    data = pd.DataFrame(columns=desired_columns)
    
        # Map values from the original DataFrame to the new DataFrame
    data['Loan_Status_Y'] = [0]
    data['ApplicantIncome'] = [float(raw_data_df['ApplicantIncome'].iloc[0])]
    data['CoapplicantIncome'] = [float(raw_data_df['CoapplicantIncome'].iloc[0])]
    data['LoanAmount'] = [float(raw_data_df['LoanAmount'].iloc[0])]
    data['Loan_Amount_Term'] = [float(raw_data_df['Loan_Amount_Term'].iloc[0])]
    data['Credit_History'] = [float(raw_data_df['Credit_History'].iloc[0])]
    data['Gender_Male'] = [1.0 if raw_data_df['Gender'].iloc[0] == 'Male' else 0.0]
    data['Married_Yes'] = [1.0 if raw_data_df['Married'].iloc[0] == 'Yes' else 0.0]
    data['Dependents_0'] = [1.0 if int(raw_data_df['Dependents'].iloc[0]) == 0 else 0.0]
    data['Dependents_1'] = [1.0 if int(raw_data_df['Dependents'].iloc[0]) == 1 else 0.0]
    data['Dependents_2'] = [1.0 if int(raw_data_df['Dependents'].iloc[0]) == 2 else 0.0]
    data['Dependents_3+'] = [1.0 if int(raw_data_df['Dependents'].iloc[0]) >= 3 else 0.0]
    data['Education_Graduate'] = [1.0 if raw_data_df['Education'].iloc[0] == 'Graduate' else 0.0]
    data['Education_Not Graduate'] = [1.0 if raw_data_df['Education'].iloc[0] == 'Not Graduate' else 0.0]
    data['Self_Employed_No'] = [1.0 if raw_data_df['Self_Employed'].iloc[0] == 'No' else 0.0]
    data['Self_Employed_Yes'] = [1.0 if raw_data_df['Self_Employed'].iloc[0] == 'Yes' else 0.0]
    data['Property_Area_Rural'] = [1.0 if raw_data_df['Property_Area'].iloc[0] == 'Rural' else 0.0]
    data['Property_Area_Semiurban'] = [1.0 if raw_data_df['Property_Area'].iloc[0] == 'Semiurban' else 0.0]
    data['Property_Area_Urban'] = [1.0 if raw_data_df['Property_Area'].iloc[0] == 'Urban' else 0.0]
    
    # Scale data
    scaler = joblib.load('./steps/scaler.pkl')
    scaled_data = pd.DataFrame(scaler.transform(data), columns=data.columns).drop('Loan_Status_Y',axis=1)
    
    return scaled_data