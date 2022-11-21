#!/usr/bin/env python
# coding: utf-8

# In[8]:


import streamlit as st


from input import image_input,video_input
from webcam import app_object_detection


st.set_page_config(layout="wide")
remove_padding_css = """
    .block-container {
    padding: 0 1rem;
    }
    """
st.markdown(
    "<style>"
    + remove_padding_css
    + "</styles>",
    unsafe_allow_html=True,
)
st.expander("foo")

st.markdown("<h1 style='text-align: left; color: black;font-size: 100px'>PPE Detection</h1>", unsafe_allow_html=True)

st.image("./data/cover.jpg",width=1500)


st.sidebar.title('Options')
method = st.sidebar.radio('Go To ->', options=['Webcam', 'Image'])

if method == 'Image':
    image_input()
# elif method == "Video":
#     video_input()

else:
    app_object_detection()






