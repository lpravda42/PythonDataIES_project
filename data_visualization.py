import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from steps.data_visualization_funcs import *

df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")

for i in ["Gender", "Married", "Education", "Dependents", "Self_Employed", "Property_Area"]:
    plot_pie_chart(df, i)

for i in ["Gender", "Married", "Education", "Dependents", "Self_Employed", "Property_Area"]:
    plot_stacked_bar_chart(df, i)

plot_scatter_plot(df, "ApplicantIncome", "CoapplicantIncome")
plot_scatter_plot(df, "LoanAmount", "ApplicantIncome")
plot_histogram(df, "ApplicantIncome")
plot_violin_plot(df, "Loan_Status", "ApplicantIncome")
plot_violin_plot(df, "Loan_Status", "LoanAmount")
plot_box_plot(df, "Loan_Status", "ApplicantIncome")