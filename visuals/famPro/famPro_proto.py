
import streamlit as st

import numpy as np 
import pandas as pd 

import df_cleaner

st.title("FAMPro")


# file = "https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-a/main/All_data_with_exits.csv"
file = "/Users/kellycho/Desktop/Repos/family-promise-spokane-ds-b/visuals/famPro/cleanedData02.csv"
df = pd.read_csv(file)



# cleaner = df_cleaner.Cleaner()
# renamed_df = cleaner.renameColumns(df)
# new_df = cleaner.set_dtypes(renamed_df)
# cleaned_df = cleaner.recategorization(new_df)

st.write(df.head())



