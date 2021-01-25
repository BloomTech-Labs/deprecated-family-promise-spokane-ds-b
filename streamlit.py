import streamlit as st
import pandas as pd
import numpy as np
#from Data Exploration.ipynb import df_pipeline3


df = pd.read_csv('/mnt/c/Users/USER/Documents/GitHub/familypromise/family-promise-spokane-ds-b/All_data_with_exits.csv')
#df = df_pipeline3
st.title("Streamlit Test")

st.header("This will be a dashboard for visualizations")
st.subheader("Visualization A")

st.markdown("## markdown")
st.text("Description text goes here")
st.success("successful")
st.info("info")
st.warning("warning")
st.error("error")
st.exception("exeception")
st.text("write with text")
st.write(range(10))

#images 
#st.image("image.jpg", width=, caption="caption here")

#videos
#vid file (example.mp4, rb)
#vid byes = vid_file.read()
#st.video(vid_file)

#Widget
# Checkbox
if st.checkbox("Show/Hide"):
    st.text("showing or Hiding Widget")

#Radio

status = st.radio("What is your status", ("Active", "Inactive"))

if status == 'Active':
    st.success("You are Active")
else:
    st.warning("Inactive, Activate")

# SelectBox
values = df['5.8 Personal ID'].tolist()
options = df['Age at Enrollment'].tolist()
dic = dict(zip(options, values))

a = st.selectbox('Personal ID', options, format_func = lambda x: dic[x])
st.write(f"{a} years old")

# MultiSelect
location = st.multiselect("Where do you work?", ("London", "New York", "Accra", "Kiev", "Nepal"))
st.write("You selected", len(location), "locations")

#Slider

level = st.slider("What is your level",1,5)

# Buttons
st.button("Simple Button")

if st.button("About"):
    st.text("Streamlit is Cool")

# Text input 
firstname = st.text_input("Enter Your Firstname", "Type Here..")
if st.button("Submit"):
    result = firstname.title()
    st.success(result)

# Date Input
import datetime
today = st.date_input("Today is", datetime.datetime.now())

# Time
the_time = st.time_input("The time is ", datetime.time())

#Displaying JSON

st.text("Display JSON")
st.json({'name':"Jesse",'gender':"male"})

#Display Raw Code

st.text("Display Raw Code")
st.code("import numpy as np")

# Display Raw Code

# with st.echo():
    #This will also show as a comment
    # import pandas as pd
    # df = pd.DataFrame()

# Progress Bar
import time
my_bar = st.progress(0)
for p in range(10):
    my_bar.progress(p + 1)


# Spinner

with st.spinner("Waiting .."):
    time.sleep(5)
st.success("Finished!")

# Balloons

#st.balloons()

# Sidebars

st.sidebar.header("About")
st.sidebar.text("This is Streamlit Tutorial")


# Functions
@st.cache
def run_fxn():
    return range(100)

st.write(run_fxn())

# Plot

#st.pyplot()

# DataFrames
st.dataframe(df)

# Tables

#st.table(df)