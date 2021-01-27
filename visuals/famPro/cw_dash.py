import streamlit as st
import awesome_streamlit as ast
import pandas as pd
from datetime import datetime


#modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from category_encoders import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import DMatrix

# plots
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# model interpretation 
import eli5
from eli5.sklearn import PermutationImportance
#from pdpbox import pdp
import shap

# images
from PIL import Image

# Data Analysis Imports
import df_cleaner

# Page Imports
import desc_stats
import lg_streamlit_app
import home


PAGES = {
    "Home" : home,
    "Descriptive Statistics" : desc_stats,
    "Machine Learning" : lg_streamlit_app,
}


st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and image 
img = Image.open("/mnt/c/Users/USER/Documents/GitHub/familypromise/family-promise-spokane-ds-b/visuals/famPro/Assets/fp_logo.png")
st.image(img, width= 900)





def main():
    ################################################
    # Select Page from navigation
    ################################################
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]
    # creates a loading message for each page
    # also takes page value and writes it in streamlit format
    with st.spinner(f"loading {selection} ..."):
        ast.shared.components.write_page(page)


if __name__ == "__main__":
    main()


