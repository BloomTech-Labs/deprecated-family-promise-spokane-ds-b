# Setup:
import awesome_streamlit as ast
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
img = Image.open(requests.get("https://raw.githubusercontent.com/Lambda-School-Labs/family-promise-spokane-ds-b/main/visuals/dashboard_app/Assets/fp_logo.png", stream=True).raw)
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
