# Setup:
from datetime import datetime
import numpy as np
import os
import pandas as pd
import streamlit as st

# Plot:
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


def write():
    @st.cache
    def loadDataFile(filePath):
        df = pd.read_csv(filePath, parse_dates=["Enroll Date", "Exit Date"])
        return df

    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, "../csv_files/des_cleaned_data.csv")

    df = loadDataFile(my_file)
    st.title("Descriptive Statistics")
    minDate = list(df["Enroll Date"].sort_values(axis=0, ascending=True).head())[0].to_pydatetime()
    maxDate = list(df["Enroll Date"].sort_values(axis=0, ascending=True).tail())[-1].to_pydatetime()
    startVal = list(df["Enroll Date"].sort_values(axis=0, ascending=True))[200].to_pydatetime()

    start_time = st.sidebar.slider(label="Date Range Start",
                                   min_value=minDate,
                                   max_value=maxDate,
                                   format="MM/DD/YY")

    end_time = st.sidebar.slider(label="Date Range End",
                                 min_value=minDate,
                                 max_value=maxDate,
                                 value=startVal,
                                 format="MM/DD/YY")

    st.write("Analyzing Range: ", start_time, "to ", end_time)

    # con_noExit = df["Exit Date"].isnull() == True
    # noExit_cleaned_df = df[con_noExit]
    # currEnrolled_df = df[con_noExit]

    con_dtRange = (df["Enroll Date"] >= start_time) & (df["Enroll Date"] <= end_time)
    currEnrolled_df = df[con_dtRange]

    # 3 different dfs
    # - total pop
    # - head of Households
    # - dependents

    # head of Households from currEnrolled
    # or cleaned_df["Relationship to HoH"] == "Significant Other (Non-Married)"
    con_Hoh = currEnrolled_df["Relationship to HoH"] == "Self"
    hoh_cleaned_df = currEnrolled_df[con_Hoh]

    # dependents from currEnrolled
    con_dep = currEnrolled_df["Relationship to HoH"] != "Self"
    dep_cleaned_df = currEnrolled_df[con_dep]

    enrolledDataList = {"All Guests Enrolled": currEnrolled_df,
                        "Heads of Household Enrolled": hoh_cleaned_df,
                        "Dependents Enrolled": dep_cleaned_df
                        }

    featureOption = st.sidebar.selectbox("Guest Type Filter ",
                                         list(enrolledDataList.keys()))

    exitComparisonVariables = {
        "Days Enrolled in Project": "Days Enrolled in Project",
        "Race": "Race",
        "Bed Nights During Report Period": "Bed Nights During Report Period",
        "Age at Enrollment": "Age at Enrollment",
        "CaseMembers": "Case Members",
        "CaseAdults": "CaseAdults"
        }

    comparisonVariableOption = st.sidebar.selectbox("Comparision Variable ",
                                                    list(exitComparisonVariables.keys()))

    source = enrolledDataList[featureOption]

    # Interactive Bar chart (Histogram)  of Population during Enrollment Range

    @st.cache(suppress_st_warning=True)
    def drawPopBarchart(popChart):
        st.altair_chart(popChart)

    st.write("Viewing: ", featureOption)
    if st.checkbox('Show Basic Enrollment Stats', value=True):
        selection2 = alt.selection_multi(fields=["Gender"], bind="legend")

        populationBarChart = alt.Chart(source).mark_bar(size=5).encode(
            alt.X("Current Age", title="Guest Current Age"),
            alt.Y('count()', title="Guest Count", stack=None),
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

        drawPopBarchart(populationBarChart)

    # Interactive Exit Comparison Bar Chart using Facet
    def drawExitComparisonFacetChart(exitFacetChart):
        st.altair_chart(exitFacetChart)

    selection = alt.selection_multi(fields=["Gender"], bind="legend")

    exitComparisonFacetBarChart = alt.Chart(source).mark_bar(size=3).encode(
        alt.X("Household ID", title=featureOption, axis=alt.Axis(labels=False)),
        alt.Y(comparisonVariableOption, stack=None),
        color="Gender",
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
    ).add_selection(
        selection
    ).properties(
        width=200,
        height=200
    ).facet(
        column="Exit Outcomes"
    ).configure_view(
        strokeWidth=0
    ).interactive()

    drawExitComparisonFacetChart(exitComparisonFacetBarChart)

    # Basic Non-Interactive Table for Exit Outcome value counts

    # get Central Tendencies to further explain charts
    def getCentralTendencies(df):
        outcomeList = ['Unknown/Other', 'NON-Permanent Exit', 'Permanent Exit']
        Mean_Days_Enrolled = [] # create a list that will be come a series in the 
                                # base df
        Mean_BedNights_Enrolled = []
        Mean_CaseMembers_Enrolled = []
        Mean_Age_At_Enrollment = []


        for outcome in outcomeList:

            conTemp = df["Exit Outcomes"] == outcome

            currentMean = df[conTemp]["Days Enrolled in Project"].mean()
            Mean_Days_Enrolled.append(currentMean)

            currBdNightsMean = df[conTemp]["Bed Nights During Report Period"].mean()
            Mean_BedNights_Enrolled.append(currBdNightsMean)

            currCsMemMean = df[conTemp]["CaseMembers"].mean()
            Mean_CaseMembers_Enrolled.append(currCsMemMean)

            currAgeAtEnMean = df[conTemp]["Age at Enrollment"].mean()
            Mean_Age_At_Enrollment.append(currAgeAtEnMean)


        return Mean_Days_Enrolled, Mean_BedNights_Enrolled, Mean_CaseMembers_Enrolled,\
            Mean_Age_At_Enrollment


    baseChart = pd.DataFrame(source["Exit Outcomes"].value_counts())
    baseChart["Mean Days Enrolled"]= pd.Series(getCentralTendencies(source)[0], index= ['Unknown/Other', 'NON-Permanent Exit', 'Permanent Exit'])
    baseChart["Mean Bed Nights Enrolled"]= pd.Series(getCentralTendencies(source)[1], index= ['Unknown/Other', 'NON-Permanent Exit', 'Permanent Exit'])
    baseChart["Mean Case Members Enrolled"]= pd.Series(getCentralTendencies(source)[2], index= ['Unknown/Other', 'NON-Permanent Exit', 'Permanent Exit'])
    baseChart["Mean Age At Enrollment"]= pd.Series(getCentralTendencies(source)[3], index= ['Unknown/Other', 'NON-Permanent Exit', 'Permanent Exit'])
    
    
    st.write(baseChart.T)
