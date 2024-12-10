import streamlit as st

st.set_page_config(
    page_title="PaperViz",
    page_icon="📖",
)

st.write("# Welcome to PaperViz!")

st.markdown(
    """
    #### Overview of our Database
"""
)

st.page_link('pages/9_Organization.py', label='Top Organization')
st.page_link('pages/10_Country.py', label='Top Country')
st.page_link('pages/11_Keyword.py', label='Top Keyword')

st.markdown(
    """
    #### Author Tools
"""
)

st.page_link('pages/12_🤵_Network_Graph.py', label='🤵 Network Graph')
st.page_link('pages/13_🌏_Geographical_Data.py', label='🌏 Geographical Data')