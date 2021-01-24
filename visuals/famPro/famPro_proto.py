
import streamlit as st

import numpy as np 
import pandas as pd 

import df_cleaner

st.title("FAMPro")


file = "https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-a/main/All_data_with_exits.csv"
df = pd.read_csv(file)

st.write(df.head())

cleaner = df_cleaner.Cleaner()
renamed_df = cleaner.renameColumns(df)
new_df = cleaner.set_dtypes(renamed_df)
cleaned_df = cleaner.recategorization(new_df)

st.write(cleaned_df.head())



# Pipeline Application
# cleaned_df = df.pipe(start_pipeline).pipe(set_dtypes).pipe(recategorization)
# Prints Resuls to show that the functions worked. 
# print(cleaned_df['Enroll Date'].dtypes)
# print(cleaned_df['Exit Date'].dtypes)
# print(cleaned_df['CurrentDate'].dtypes)
# print(cleaned_df['Date of First Contact (Beta)'].dtypes)
# print(cleaned_df['Date of First ES Stay (Beta)'].dtypes)
# print(cleaned_df['Date of Last Contact (Beta)'].dtypes)
# print(cleaned_df['Date of Last ES Stay (Beta)'].dtypes)
# print(cleaned_df['Engagement Date'].dtypes)
# print(cleaned_df['Homeless Start Date'].dtypes)
# print(cleaned_df['Recategorized'].value_counts(dropna=False))