import streamlit as st

from joblib import load
from pdpbox.pdp import pdp_isolate, pdp_plot

import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "randomforest_model.joblib")
model = load(my_file)

import pandas as pd
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "All_data_with_exits_cleaned.csv")
df = pd.read_csv(my_file)

target = 'Target Exit Destination'
X = df.drop(columns=[target])
y = df[target]

from sklearn.model_selection import train_test_split
# Train, Test, Validation Split
# First split : Train, Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Second split : Train, Val
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)


# feature = "CaseMembers"
# isolated = pdp_isolate(
#     model=model,
#     dataset=X_val,
#     model_features=X_val.columns,
#     feature = feature
# )

# st.write(pdp_plot(isolated[:1], feature_name=feature))

chart_data = df[['Age at Enrollment', 'Race']][:100]
st.area_chart(chart_data)