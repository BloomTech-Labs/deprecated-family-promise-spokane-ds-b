"""Data visualization functions
   """
# Setup:
import eli5
import joblib
import numpy as np
import os
import pandas as pd
import streamlit as st

# Plot:
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning:
from catboost import CatBoostClassifier
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Interpretation:
from eli5 import explain_weights_df, show_weights
from eli5.sklearn import PermutationImportance
from pdpbox.pdp import pdp_isolate, pdp_plot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix, classification_report
import shap
from fastapi import APIRouter
from fps_dashboard import des_statistics
router = APIRouter()
@router.post('/draw_pop_chart')
async def pop_bar_chart(popChart):
   return {'Bar Chart': des_statistics.drawPopBarchart(populationBarChart)}

   
@router.post('/visualization')
async def drawPopBarchart(popChart):
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

      return {'Bar Chart': drawPopBarchart(populationBarChart)}


# Interactive Exit Comparison Bar Chart using Facet
@router.post('/Visualization')
async def drawExitComparisonFacetChart(exitFacetChart):
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

   return {'Comparison Face tChart': drawExitComparisonFacetChart(exitComparisonFacetBarChart)}


@router.post('/Visualization')
async def make_class_metrics(target, pred, training_set, model, ml_name):

   """show model performance metrics such as classification report and
   confusion matrix"""
   # Classification report
   report = classification_report(target, pred, output_dict=True)
   st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())
   # Confusion matrix
   st.sidebar.markdown("#### Confusion Matrix")
   fig, ax = plt.subplots()
   plot_confusion_matrix(model, training_set, target,
                        normalize='true', xticks_rotation='vertical', ax=ax)
   ax.set_title((f'{ml_name} Confusion Matrix'), fontsize=10,
               fontweight='bold')
   ax.grid(False)
   return {'plot': st.sidebar.pyplot(fig=fig, clear_figure=True)}



