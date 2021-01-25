import streamlit as st
import pandas as pd

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

# model interpretation 
import eli5
from eli5.sklearn import PermutationImportance
#from pdpbox import pdp
import shap

# images
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and image 
img = Image.open("/mnt/c/Users/USER/Documents/GitHub/familypromise/family-promise-spokane-ds-b/viz-dash/assets/fp_logo.png")
st.image(img, width= 900)
st.title("Data Vizualitations Dashboard")

#uploaded_file = '/mnt/c/Users/USER/Documents/GitHub/familypromise/family-promise-spokane-ds-b/viz-dash/All_data_with_exits_cleaned.csv'
#uploading dataframe for model interpretation
def upload_data(uploaded_file):
    if uploaded_file is not None:
        st.sidebar.success("File uploaded!")
        df = pd.read_csv(uploaded_file, encoding="utf8")
        # replace all non alphanumeric column names to avoid lgbm issue
        df.columns = [
            "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
        ]
        # make the last col the default outcome
        col_arranged = df.columns[:-1].insert(0, df.columns[-1])
        target_col = st.sidebar.selectbox(
            "Then choose the target variable", col_arranged
        )
        X, y, features, target_labels = encode_data(df, target_col)
    return df, X, y, features, target_labels

def encode_data(data, targetcol):
    """preprocess categorical value"""
    X = pd.get_dummies(data.drop(targetcol, axis=1)).fillna(0)
    X.columns = ["".join(c if c.isalnum() else "_" for c in str(x)) for x in X.columns]
    features = X.columns
    data[targetcol] = data[targetcol].astype("object")
    target_labels = data[targetcol].unique()
    y = pd.factorize(data[targetcol])[0]
    return X, y, features, target_labels

def splitdata(X, y):
    """split dataset into trianing & testing"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.80, random_state=0
    )
    return X_train, X_test, y_train, y_test

def make_pred(dim_model, X_test, clf):
    """get y_pred using the classifier"""
    if dim_model == "XGBoost":
        pred = clf.predict(DMatrix(X_test))
    return pred

def show_global_interpretation_eli5(X_train, y_train, features, clf, dim_model):
    """show most important features via permutation importance in ELI5"""
    if dim_model == "XGBoost":
        df_global_explain = eli5.explain_weights_df(
            clf, feature_names=features.values, top=5
        ).round(2)
    else:
        perm = PermutationImportance(clf, n_iter=2, random_state=1).fit(
            X_train, y_train
        )
        df_global_explain = eli5.explain_weights_df(
            perm, feature_names=features.values, top=5
        ).round(2)
    bar = (
        alt.Chart(df_global_explain)
        .mark_bar(color="red", opacity=0.6, size=16)
        .encode(x="weight", y=alt.Y("feature", sort="-x"), tooltip=["weight"])
        .properties(height=160)
    )
    st.write(bar)

def show_local_interpretation_shap(clf, X_test, pred, slider_idx):
    """show the interpretation of individual decision points"""



    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)
    # the predicted class for the selected instance
    pred_i = int(pred[slider_idx])
    # this illustrates why the model predict this particular outcome
    shap.force_plot(
        explainer.expected_value[pred_i],
        shap_values[pred_i][slider_idx, :],
        X_test.iloc[slider_idx, :],
        matplotlib=True,
    )
    st.pyplot()

def show_local_interpretation(
    X_test, y_test, clf, pred, target_labels, features, dim_model
):
    """show the interpretation based on the selected framework"""
    n_data = X_test.shape[0]
    slider_idx = st.slider("Which datapoint to explain", 0, n_data - 1)

    st.text(
        "Prediction: "
        + str(target_labels[int(pred[slider_idx])])
        + " | Actual label: "
        + str(target_labels[int(y_test[slider_idx])])
    )

    show_local_interpretation_shap(clf, X_test, pred, slider_idx)


def show_global_interpretation_shap(X_train, clf):
    """show most important features via permutation importance in SHAP"""
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        max_display=5,
        plot_size=(12, 5),
        color=plt.get_cmap("tab20b"),
        show=False,
        color_bar=False,
    )
    # note: there might be figure cutoff issue. Will look further into forceplot & st.pyplot's implementation.
    st.pyplot()



def draw_pdp(clf, dataset, features, target_labels, dim_model):
    """draw pdpplot given a model, data, all the features and the selected feature to plot"""

    if dim_model != "XGBoost":
        selected_col = st.selectbox("Select a feature", features)
        st.info(
            """**To read the chart:** The curves describe how a feature marginally varies with the likelihood of outcome. Each subplot belong to a class outcome.
        When a curve is below 0, the data is unlikely to belong to that class.
        [Read more] ("https://christophm.github.io/interpretable-ml-book/pdp.html") """
        )

        pdp_dist = pdp.pdp_isolate(
            model=clf, dataset=dataset, model_features=features, feature=selected_col
        )
        if len(target_labels) <= 5:
            ncol = len(target_labels)
        else:
            ncol = 5
        pdp.pdp_plot(pdp_dist, selected_col, ncols=ncol, figsize=(12, 5))
        st.pyplot()


def main():
    ################################################
    # upload file
    ################################################

    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    df, X, y, features, target_labels = upload_data(uploaded_file)

    ################################################
    # process data
    ################################################

    X_train, X_test, y_train, y_test = splitdata(X, y)

    ################################################
    # apply model
    ################################################
    dim_model = st.sidebar.selectbox(
        "Choose a model", ("XGBoost", "randomforest")
    )
    if dim_model == "randomforest":
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1)
        clf.fit(X_train, y_train)

    elif dim_model == "XGBoost":
        params = {
            "max_depth": 5,
            "silent": 1,
            "random_state": 2,
            "num_class": len(target_labels),
        }
        dmatrix = DMatrix(data=X_train, label=y_train)
        clf = xgb.train(params=params, dtrain=dmatrix)

    ################################################
    # Predict
    ################################################
    pred = make_pred(dim_model, X_test, clf)



    ################################################
    # Model output
    ################################################
 

    # the report is formatted to 2 decimal points (i.e. accuracy 1 means 1.00) dependent on streamlit styling update https://github.com/streamlit/streamlit/issues/1125
 
    

    ################################################
    # Global Interpretation
    ################################################
    st.markdown("#### Feature Importance")
    info_global = st.button("Explanation")
    if info_global:
        st.info(
            """
        Explanatory data analytics here......
        """
        )
    # This only works if removing newline from html
    # Refactor this once added more models
    show_global_interpretation_shap(X_train, clf)


    if st.sidebar.button("About the app"):
        st.sidebar.markdown(
            """
             Read more about how it works on [Github] (https://github.com/yanhann10/ml_interpret)
             Basic data cleaning recommended before upload   
             [Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdTXKpMPC0-TmWf2ngU9A0sokH5Z0m-QazSPBIZyZ2AbXIBug/viewform?usp=sf_link)   
             Last update Mar 2020 by [@hannahyan](https://twitter.com/hannahyan)
              """
        )
        st.sidebar.markdown(
            '<a href="https://ctt.ac/zu8S4"><img src="https://image.flaticon.com/icons/svg/733/733579.svg" width=16></a>',
            unsafe_allow_html=True,
        )

    ################################################
    # Local Interpretation
    ################################################
    st.markdown("#### Local Interpretation")

    # misclassified
    if st.checkbox("Filter for misclassified"):
        X_test, y_test, pred = filter_misclassified(X_test, y_test, pred)
        if X_test.shape[0] == 0:
            st.text("No misclassification🎉")
        else:
            st.text(str(X_test.shape[0]) + " misclassified total")
            show_local_interpretation(
                X_test,
                y_test,
                clf,
                pred,
                target_labels,
                features,
                dim_model,
                dim_framework,
            )
    else:
        show_local_interpretation(
            X_test, y_test, clf, pred, target_labels, features, dim_model,
        )

    ################################################
    # PDP plot
    ################################################
    if dim_model != "XGBoost" and st.checkbox("Show how features vary with outcome"):
        draw_pdp(clf, X_train, features, target_labels, dim_model)


if __name__ == "__main__":
    main()



