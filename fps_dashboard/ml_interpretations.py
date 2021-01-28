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

# To disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# This will set the style for all matplots
plt.style.use('classic')

# """Not in use at the moment"""
# Load Models
# cat_model = joblib.load('assets/CatBoost_Model.joblib')
# xgb_model = joblib.load('assets/XGBoost_Model.joblib')
# forest_model = joblib.load('assets/Forest_Model.joblib')


def upload_data(uploaded_file):
    """To process the csv file in order to return training data"""
    if uploaded_file is not None:
        # """Not in use at the moment"""
        # st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, index_col='Personal ID',
                         encoding="utf8")
        col_names = df.columns[:-1].insert(0, df.columns[-1])
        # Dataset preview if selected
        st.sidebar.markdown("#### To display dataset")
        if st.sidebar.checkbox("Preview uploaded data"):
            st.dataframe(df.head())
        # Target column is selected by default
        target_cols = st.sidebar.selectbox(
            "Target variable", col_names[:1]
        )
        X = df.drop(target_cols, axis=1)
        y = df[target_cols]
        return X, y, df, target_cols


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


def process_data(X_train, X_test, X_val, X):
    """pre-process training data and transformation"""
    processor = make_pipeline(OrdinalEncoder(), SimpleImputer())
    X_train = processor.fit_transform(X_train)
    X_val = processor.transform(X_val)
    X_test = processor.transform(X_test)
    encoded_cols = list(range(0, X.shape[1]))
    column_names = list(X.columns)
    features = dict(zip(encoded_cols, column_names))
    return X_train, X_test, X_val, features, column_names, processor


def make_prediction(training_set, model):
    """to get y_pred"""
    pred = model.predict(training_set)
    return pred


def make_class_metrics(target, pred, training_set, model, ml_name):
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
    st.sidebar.pyplot(fig=fig, clear_figure=True)


def make_eli5_interpretation(training_set, target, model,
                             features, X, ml_name):
    """to display most important features via permutation in eli5
    and sklearn formats"""
    # Permutation importances by eli5
    perm = PermutationImportance(model, n_iter=1,
                                 random_state=0).fit(training_set, target)
    df_explain = explain_weights_df(perm,
                                    feature_names=features, top=10).round(3)
    bar = (
        alt.Chart(df_explain, title=f'ELI5 Weights Explained from {ml_name}')
        .mark_bar(color="red", opacity=0.6, size=14)
        .encode(x="weight", y=alt.Y("feature", sort="-x"), tooltip=["weight"])
        .properties(height=300, width=675)
    )
    st.markdown("#### Feature Importance Interpretation")
    info_global = st.button("How it is calculated")
    st.write(bar)

    st.markdown("#### Local Interpretation")
    info_local = st.button("How this works")
    # Permutation importances by sklearn
    imp = permutation_importance(model, training_set, target, random_state=0)

    data = {'importances_mean': imp['importances_mean'],
            'importances_std': imp['importances_std']}
    imp = pd.DataFrame(data, index=X.columns)
    imp.sort_values('importances_mean', ascending=False, inplace=True)

    fig, ax = plt.subplots(figsize=(12, 16))
    imp.importances_mean.plot(kind='barh', ax=ax)
    plt.title('Sklearn Permutation Importances', fontsize=14,
              fontweight='bold')
    plt.xlabel(ml_name, fontsize=12)

    plt.tight_layout()
    st.write(fig)


def make_pdp_interpretation(dataset, column_names, training_set, model):
    """to display partial dependence plots based on user input"""
    X_pdp = pd.DataFrame(training_set, columns=column_names)
    col_pdp = st.selectbox(
            "Choose the feature to plot", column_names
    )
    feature = col_pdp
    class_list = list(dataset['Target Exit Destination'].value_counts().index)
    target_value = st.selectbox(
        "Choose the class to plot", class_list, index=1
    )
    isolated = pdp_isolate(
        model=model,
        dataset=X_pdp,
        model_features=X_pdp.columns,
        feature=feature,
    )
    if target_value == 'Unknown/Other':
        pdp_plot(isolated[0], feature_name=[feature, target_value])
    elif target_value == 'Permanent Exit':
        pdp_plot(isolated[1], feature_name=[feature, target_value])
    elif target_value == 'Emergency Shelter':
        pdp_plot(isolated[2], feature_name=[feature, target_value])
    elif target_value == 'Temporary Exit':
        pdp_plot(isolated[3], feature_name=[feature, target_value])
    elif target_value == 'Transitional Housing':
        pdp_plot(isolated[4], feature_name=[feature, target_value])
    st.pyplot()
    st.markdown("#### Feature Importance Interpretation")
    info_global = st.button("How it is calculated")


def make_shap_interpretation(model, training_set, column_names, ml_name,
                             target, dataset, X, processor):
    """display shap's multi class values and force plots based on
    personal id selection"""
    # Summary plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(training_set)
    shap.summary_plot(shap_values, column_names,
                      class_names=model.classes_, plot_type='bar',
                      max_display=10, show=True, auto_size_plot=True)
    plt.title(f'SHAP Multi Class Values from {ml_name}',
              fontsize=12, fontweight='bold')
    plt.legend(loc='lower right')
    st.markdown("#### Feature Importance Interpretation")
    info_global = st.button("How it is calculated")
    st.pyplot()

    st.markdown("#### Local Interpretation")
    info_local = st.button("How this works")
    # Force plot
    slider_idx = st.selectbox('Personal ID of Guest', X.index)
    row_p = X.loc[[slider_idx]]
    row = processor.transform(row_p)
    explainer_force = shap.TreeExplainer(model)
    shap_values_force = explainer_force.shap_values(row)

    class_list = list(dataset['Target Exit Destination'].value_counts().index)
    target_value = st.selectbox(
        "Choose the class to plot", class_list, index=1
    )
    shap.initjs()
    if target_value == 'Unknown/Other':
        shap.force_plot(
            base_value=explainer_force.expected_value[0],
            shap_values=shap_values_force[0],
            features=row,
            feature_names=column_names,
            link='logit'
        ).matplotlib(figsize=(20, 8), show=True, text_rotation=45)
    elif target_value == 'Permanent Exit':
        shap.force_plot(
            base_value=explainer_force.expected_value[1],
            shap_values=shap_values_force[1],
            features=row,
            feature_names=column_names,
            link='logit'
        ).matplotlib(figsize=(20, 8), show=True, text_rotation=45)
    elif target_value == 'Emergency Shelter':
        shap.force_plot(
            base_value=explainer_force.expected_value[2],
            shap_values=shap_values_force[2],
            features=row,
            feature_names=column_names,
            link='logit'
        ).matplotlib(figsize=(20, 8), show=True, text_rotation=45)
    elif target_value == 'Temporary Exit':
        shap.force_plot(
            base_value=explainer_force.expected_value[3],
            shap_values=shap_values_force[3],
            features=row,
            feature_names=column_names,
            link='logit'
        ).matplotlib(figsize=(20, 8), show=True, text_rotation=45)
    elif target_value == 'Transitional Housing':
        shap.force_plot(
            base_value=explainer_force.expected_value[4],
            shap_values=shap_values_force[4],
            features=row,
            feature_names=column_names,
            link='logit'
        ).matplotlib(figsize=(20, 8), show=True, text_rotation=45)
    st.pyplot()


def write():
    # Title and Subheader
    st.title("Machine Learning Interpretation")
    st.subheader("Family Promise of Spokane")
    # CSV File Upload
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, "../csv_files/ml_cleaned_data.csv")
    X, y, df, target_cols = upload_data(my_file)

    # Split Data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    # Process Training Data
    X_train, X_test, X_val, features, column_names, processor = process_data(
        X_train, X_test, X_val, X)

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

    # Prediction Data Selection
    sets = st.sidebar.selectbox(
        "Choose a set", ("Test 20%", "Validation 20%")
    )

    # Interpretation Framework Selection
    framework = st.sidebar.radio(
        "Choose interpretation framework", ["ELI5 + Permutation Importances",
                                            "PDP", "SHAP"]
    )

    # Title classification report
    st.sidebar.markdown("#### Classification report")

    # Interpretations
    if sets == "Test 20%":
        # To get y pred
        pred = make_prediction(X_test, model)
        # To display classification report and confusion matrix
        make_class_metrics(y_test, pred, X_test, model, ml_name)
        if framework == "ELI5 + Permutation Importances":
            # To display eli5 weights and permutation importances
            make_eli5_interpretation(X_test, y_test, model,
                                     features, X, ml_name)
        elif framework == "PDP":
            # To display pdp isolated plots
            make_pdp_interpretation(df, column_names, X_test, model)
        elif framework == "SHAP":
            # To display shap summary and force plots
            make_shap_interpretation(model, X_test, column_names, ml_name,
                                     y_test, df, X, processor)
    elif sets == "Validation 20%":
        pred = make_prediction(X_val, model)
        make_class_metrics(y_val, pred, X_val, model, ml_name)
        if framework == "ELI5 + Permutation Importances":
            make_eli5_interpretation(X_val, y_val, model, features, X, ml_name)
        elif framework == "PDP":
            make_pdp_interpretation(df, column_names, X_val, model)
        elif framework == "SHAP":
            make_shap_interpretation(model, X_val, column_names, ml_name,
                                     y_val, df, X, processor)
