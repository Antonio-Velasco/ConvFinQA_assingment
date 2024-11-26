import streamlit as st

APP_TITLE = "Tomoro Technical Assigment"
APP_ICON = "⚙️"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
)

st.write("# Welcome! 👋")

st.sidebar.success("Select a tool above.")

st.markdown(
    """
    **👈 Select a tool from the sidebar to start**

    # Overview

    ## Powered by Streamlit.
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    ##### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
"""
)
