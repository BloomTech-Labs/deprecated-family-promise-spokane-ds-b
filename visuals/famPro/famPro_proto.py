
import streamlit as st

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
import df_cleaner

# st.title("FAMPro")
""" # FAMPro
    ##### F`amily Promise`   A`nalytics`   M`ode`    Pro`totype`
"""


file = "https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-b/ergVizExitVariable/visuals/famPro/cleanedData02.csv"
df = pd.read_csv(file, parse_dates=["Enroll Date", "Exit Date"])



# cleaner = df_cleaner.Cleaner()
# renamed_df = cleaner.renameColumns(df)
# new_df = cleaner.set_dtypes(renamed_df)
# cleaned_df = cleaner.recategorization(new_df)

# st.write(df.head())

# Seaborn viz


# fig = sns.displot(df, x="Current Age",  hue="Gender", bins=20, multiple = "stack")

# st.pyplot(fig)

from datetime import datetime

minDate = list(df["Enroll Date"].sort_values(axis=0, ascending=True).head())[0].to_pydatetime()
maxDate = list(df["Enroll Date"].sort_values(axis=0, ascending=True).tail())[-1].to_pydatetime()
"""


"""
start_time = st.sidebar.slider(label = "Date Range Start",
            min_value=minDate,
            max_value=maxDate,
            format="MM/DD/YY")

end_time = st.sidebar.slider(label = "Date Range End",
            min_value=minDate,
            max_value=maxDate,
            format="MM/DD/YY")


st.write("Analyzing Range: ", start_time, "to ", end_time)








# altair viz
import altair as alt

con_noExit = df["Exit Date"].isnull() == True
# noExit_cleaned_df = df[con_noExit]
# currEnrolled_df = df[con_noExit]

con_dtRange = (df["Enroll Date"] >= start_time) & (df["Enroll Date"] <= end_time)
currEnrolled_df = df[con_dtRange]
# con_dtGreater2017 = cleaned_df["Enroll Date"] > pd.Timestamp(2019,12, 31)
# cleaned_df[con_dtGreater2017].shape



# Non-Interactive chart
# test = alt.Chart(source).mark_bar().encode(
#     alt.Y("Current Age", title = "Current Guest Age"),
#     alt.X('count()'),
#     color="Gender",
   
# )

# 3 different dfs
# total pop
# head of Households
# dependents

# Currently enrolled 
# con_noExit = df["Exit Date"].isnull() == True
# currEnrolled_df = df[con_noExit]

# head of Households from currEnrolled
con_Hoh = currEnrolled_df["Relationship to HoH"] == "Self" #or cleaned_df["Relationship to HoH"] == "Significant Other (Non-Married)"
hoh_cleaned_df = currEnrolled_df[con_Hoh]

# dependents from currEnrolled
con_dep = currEnrolled_df["Relationship to HoH"] != "Self"
dep_cleaned_df = currEnrolled_df[con_dep]

enrolledDataList = {"All Guests Enrolled": currEnrolled_df, 
                    "Heads of Household Enrolled": hoh_cleaned_df,
                    "Dependents Enrolled": dep_cleaned_df
}

featureOption = st.sidebar.selectbox("Feature ", 
    list(enrolledDataList.keys()))

"Viewing: ", featureOption

# st.altair_chart(test)

# source =currEnrolled_df

source = enrolledDataList[featureOption]


# Interactive test
selection = alt.selection_multi(fields=["Gender"], bind="legend")

test2 = alt.Chart(source).mark_bar().encode(
    alt.X("Current Age", title = "Current Guest Age"),
    alt.Y('count()', stack = None),
    color="Gender",
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).add_selection(
    selection
).properties(width=550).interactive()

st.altair_chart(test2)

