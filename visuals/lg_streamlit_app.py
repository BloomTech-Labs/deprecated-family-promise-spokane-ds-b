# Setup:
import eli5
import joblib
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

# Interpretation:
from eli5 import show_weights
from pdpbox.pdp import pdp_isolate, pdp_plot
from sklearn.inspection import permutation_importance
from sklearn.metrics import plot_confusion_matrix, classification_report
import shap


# """Not in use at the moment"""
# Load Models
# cat_model = joblib.load('visuals/CatBoost_Model.joblib')
# xgb_model = joblib.load('visuals/XGBoost_Model.joblib')
# forest_model = joblib.load('visuals/Forest_Model.joblib')

# Title and Subheader
st.title("Machine Learning Interpretation")
st.subheader("Family Promise of Spokane")


def upload_data(uploaded_file):
    """To process the csv file in order to return training data"""
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
        column_names = df.columns[:-1].insert(0, df.columns[-1])
        target = st.sidebar.selectbox(
            "Choose the target varible", column_names
        )
        X = df.drop(target, axis=1)
        y = df[target]
        return X, y, df


def split_data(X, y):
    """split dataset into training, validation & testing"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80,
                                                        test_size=0.20,
                                                        random_state=0)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      train_size=0.75,
                                                      test_size=0.25,
                                                      random_state=0)

    return X_train, X_test, X_val, y_train, y_test, y_val


def process_data(X_train, X_test, X_val):
    """pre-process training data and transformation"""
    processor = make_pipeline(OrdinalEncoder(), SimpleImputer())
    X_train = processor.fit_transform(X_train)
    X_val = processor.transform(X_val)
    X_test = processor.transform(X_test)

    return X_train, X_test, X_val


def make_prediction(p, model):
    """to get y_pred"""
    pred = model.predict(p)
    return pred


def make_class_metrics(t, pred, p, model, ml_name):
    """show model performance metrics such as classification report and
    confusion matrix"""
    report = classification_report(t, pred, output_dict=True)
    st.sidebar.dataframe(pd.DataFrame(report).round(1).transpose())

    st.sidebar.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots()
    plot_confusion_matrix(model, p, t,
                          normalize='true', xticks_rotation='vertical', ax=ax)
    ax.set_title((f'{ml_name} Confusion Matrix'), fontsize=10,
                 fontweight='bold')
    ax.grid(False)
    st.sidebar.pyplot(fig=fig, clear_figure=True)


def main():
    # CSV File Upload
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    X, y, df = upload_data(uploaded_file)

    # Split Data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Process Training Data
    X_train, X_test, X_val = process_data(X_train, X_test, X_val)

    # Model Selection
    ml_name = st.sidebar.selectbox(
        "Choose a model", ("CatBoost", "XGBoost", "RandomForest")
    )
    if ml_name == "CatBoost":
        model = CatBoostClassifier(iterations=100, random_state=0,
                                   verbose=0)
        model.fit(X_train, y_train)
    elif ml_name == "XGBoost":
        model = XGBClassifier(n_estimators=25, random_state=0,
                              booster='gbtree', verbosity=0)
        model.fit(X_train, y_train)
    elif ml_name == "RandomForest":
        model = RandomForestClassifier(n_estimators=25, random_state=0,
                                       verbose=0)
        model.fit(X_train, y_train)

    # Display Accuracy Scores
    st.sidebar.markdown("#### Model Accuracy")
    st.sidebar.write("Test: ", round(model.score(X_test, y_test), 3))
    st.sidebar.write("Validation: ", round(model.score(X_val, y_val), 3))

    # Prediction Data
    sets = st.sidebar.selectbox(
        "Choose a set", ("Test 20%", "Validation 20%")
    )
    st.sidebar.markdown("#### Classification report")
    if sets == "Test 20%":
        pred = make_prediction(X_test, model)
        make_class_metrics(y_test, pred, X_test, model, ml_name)
    elif sets == "Validation 20%":
        pred = make_prediction(X_val, model)
        make_class_metrics(y_val, pred, X_val, model, ml_name)

    # Interpretation Framework
    dim_framework = st.sidebar.radio(
        "Choose interpretation framework", ["SHAP", "ELI5"]
    )

    # Dataset preview if selected
    if st.sidebar.checkbox("Preview uploaded data"):
        st.dataframe(df.head())

if __name__ == "__main__":
    main()
