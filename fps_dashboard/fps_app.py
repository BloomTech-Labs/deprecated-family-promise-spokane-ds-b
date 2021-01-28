# Setup:
import awesome_streamlit as ast
import os
import streamlit as st

# Images:
from PIL import Image
import requests

# Page:
import des_statistics
import home
import ml_interpretations


PAGES = {
    "Home": home,
    "Descriptive Statistics": des_statistics,
    "Machine Learning": ml_interpretations,
}

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title and image
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file = os.path.join(THIS_FOLDER, "../fps_dashboard/Assets/fp_logo.png")
img = Image.open(my_file)
st.image(img, width=900)


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
