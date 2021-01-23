# Setup:
import eli5
import numpy as np
import pandas as pd
import streamlit as st

# Plot:
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning:
from catboost import CatBoostClassifier
from category_encoders import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Interpretation:
from eli5 import show_weights
from eli5.sklearn import PermutationImportance
from pdpbox.pdp import pdp_isolate, pdp_interact, pdp_plot, pdp_interact_plot
import shap

# Title and Subheader
st.title("Machine Learning Interpretation")
st.subheader("Family Promise of Spokane")