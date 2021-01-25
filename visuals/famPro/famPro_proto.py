
import streamlit as st

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
import df_cleaner

st.title("FAMPro")


# file = "https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-a/main/All_data_with_exits.csv"
file = "/Users/kellycho/Desktop/Repos/family-promise-spokane-ds-b/visuals/famPro/cleanedData02.csv"
df = pd.read_csv(file)



# cleaner = df_cleaner.Cleaner()
# renamed_df = cleaner.renameColumns(df)
# new_df = cleaner.set_dtypes(renamed_df)
# cleaned_df = cleaner.recategorization(new_df)


# Seaborn viz
# st.write(df.head())


fig = sns.displot(df, x="Current Age",  hue="Gender", bins=20, multiple = "stack")

st.pyplot(fig)

# 3 different dfs
# total pop
# head of Households
# dependents

# Currently enrolled 
con_noExit = df["Exit Date"].isnull() == True
currEnrolled_df = df[con_noExit]

# head of Households from currEnrolled
con_Hoh = currEnrolled_df["Relationship to HoH"] == "Self" #or cleaned_df["Relationship to HoH"] == "Significant Other (Non-Married)"
hoh_cleaned_df = currEnrolled_df[con_Hoh]

# dependents from currEnrolled
con_dep = currEnrolled_df["Relationship to HoH"] != "Self"
dep_cleaned_df = currEnrolled_df[con_dep]

enrolledDataList = {"All Currently Enrolled": currEnrolled_df, 
                    "Heads of Household Enrolled": hoh_cleaned_df,
                    "Dependents Enrolled": dep_cleaned_df
}

featureOption = st.sidebar.selectbox("Feature ", 
    list(enrolledDataList.keys()))

"Viewing: ", featureOption



# altair viz
import altair as alt

con_noExit = df["Exit Date"].isnull() == True
# noExit_cleaned_df = df[con_noExit]
currEnrolled_df = df[con_noExit]

source =currEnrolled_df

# Non-Interactive chart
# test = alt.Chart(source).mark_bar().encode(
#     alt.Y("Current Age", title = "Current Guest Age"),
#     alt.X('count()'),
#     color="Gender",
   
# )

# st.altair_chart(test)

# Interactive test
selection = alt.selection_multi(fields=["Gender"], bind="legend")

test2 = alt.Chart(source).mark_bar().encode(
    alt.X("Current Age", title = "Current Guest Age"),
    alt.Y('count()'),
    color="Gender",
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).add_selection(
    selection
).properties(width=550).interactive()

st.altair_chart(test2)

