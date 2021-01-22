# Create your first Streamlit app
import streamlit as st
import pandas as pd
import numpy as np

# Add text and data
st.title("My first app")

"""Write a data frame"""
st.write("Here's our first attempt at using data to create a table")
st.write(pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
}))

# Use Magic
"""Without using [st.write]"""
df = st.write(pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
}))

st.markdown("**Line Chart**")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=["a", "b", "c"]
)

st.line_chart(chart_data)

st.markdown("**Map Plot**")
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=["lat", "lon"]
)

st.map(map_data)

st.markdown("**Use checkboxes to show/hide data**")
if st.checkbox("Show dataframe"):
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=["a", "b", "c"]
    )

    st.line_chart(chart_data)


df = pd.DataFrame({
    "first column": [1, 2, 3, 4],
    "second column": [10, 20, 30, 40]
})

st.markdown("**Use a selectbox for options**")
option_selectbox = st.selectbox(
    "Which number do you like best?",
    df["first column"]
)

"You selected: ", option_selectbox

st.markdown("**Use a sidebar for options**")
option_sidebar = st.sidebar.selectbox(
    "Which number do you like best?",
    df["second column"]
)

"You selected:", option_sidebar

st.markdown("**Use of button option**")
left_column, right_column = st.beta_columns(2)
pressed = left_column.button("Press me?")
if pressed:
    right_column.write("Woohoo!")

expander = st.beta_expander("FAQ")
expander.write("Here you could put in some really, really long explanations...")

st.markdown("**Use of progress bar**")
import time

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
    # Update the progress bar with each iteration
    latest_iteration.text(f"Iteration {i+1}")
    bar.progress(i+1)
    time.sleep(0.1)

"...and now we're done!"

st.markdown("**Family Promise DataFrame Version 3**")
df_note = pd.read_csv("cleaned_df.csv")
st.table(df_note.head())