
import streamlit as st

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt
import df_cleaner

from datetime import datetime

""" # FAMPro
    ##### F`amily Promise`   A`nalytics`   M`ode`    Pro`totype`
"""


# file = "https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-b/ergVizExitVariable/visuals/famPro/cleanedData02.csv"
file = "cleanedData02.csv"
df = pd.read_csv(file, parse_dates=["Enroll Date", "Exit Date"])



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


# 3 different dfs
# total pop
# head of Households
# dependents


# head of Households from currEnrolled
con_Hoh = currEnrolled_df["Relationship to HoH"] == "Self" #or cleaned_df["Relationship to HoH"] == "Significant Other (Non-Married)"
hoh_cleaned_df = currEnrolled_df[con_Hoh]

# dependents from currEnrolled
con_dep = currEnrolled_df["Relationship to HoH"] != "Self"
dep_cleaned_df = currEnrolled_df[con_dep]

enrolledDataList = {"All Guests Enrolled": currEnrolled_df, 
                    "Heads of Household Enrolled": hoh_cleaned_df,
                    "Dependents Enrolled": dep_cleaned_df,
}

featureOption = st.sidebar.selectbox("Guest Type Filter ", 
    list(enrolledDataList.keys()))


exitComparisonVariables = {
    "Days Enrolled in Project" : "Days Enrolled in Project",
    "Race": "Race", 
    "Bed Nights During Report Period": "Bed Nights During Report Period",
    "Age at Enrollment" : "Age at Enrollment",
    "CaseMembers": "Case Members",
    "CaseAdults" : "CaseAdults"
}

comparisonVariableOption = st.sidebar.selectbox("Comparision Variable ", 
    list(exitComparisonVariables.keys()))



"Viewing: ", featureOption

source = enrolledDataList[featureOption]

# Interactive Bar chart (Histogram)  of Population during Enrollment Range
if st.checkbox('Show Basic Enrollment Stats'):    
    selection2 = alt.selection_multi(fields=["Gender"], bind="legend")

    populationBarChart = alt.Chart(source).mark_bar(size=5).encode(
        alt.X("Current Age", title = "Guest Current Age"),
        alt.Y('count()',title = "Guest Count", stack = None),
        color="Gender",
        opacity=alt.condition(selection2, alt.value(1), alt.value(0.2))
    ).add_selection(
        selection2
    ).properties(
        width=550
    ).configure_axis(
        grid=False
    ).configure_view(
        strokeWidth=0
    ).interactive()

    st.altair_chart(populationBarChart)

# Interactive Exit Comparison Bar Chart


selection = alt.selection_multi(fields=["Gender"], bind="legend")

exitComparisonFacetBarChart = alt.Chart(source).mark_bar(size=3).encode(
    alt.X("Household ID", title = featureOption,axis=alt.Axis(labels=False), 
    alt.Y(comparisonVariableOption,  stack = None),
    color="Gender",
    opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
).add_selection(
    selection
).properties(
    width=200,
    height=200
).facet(
    column="Descriptive Viz Category"
# ).configure_axis(
#     grid=False
).configure_view(
    strokeWidth=0
).interactive()


st.altair_chart(exitComparisonFacetBarChart)