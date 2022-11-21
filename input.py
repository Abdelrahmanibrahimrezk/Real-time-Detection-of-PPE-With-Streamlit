#!/usr/bin/env python
# coding: utf-8

# In[1]:


import threading
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings
from PIL import Image
import cv2
from detect_img import detect
from detect_video import detect_video
import streamlit as st
import cv2 as cv
import tempfile

def image_input():
    
    content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    
    if content_file is not None:
        content = Image.open(content_file)
        content = np.array(content) #pil to cv
        generated = detect(content)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
    st.image(generated)
def video_input():
    content_file = st.sidebar.file_uploader("Choose a Content Video", type=['mp4'])
    if content_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(content_file.read())
        vid = cv.VideoCapture(tfile.name)
        generated = detect_video(vid)
        st.video(generated)

    else:
        st.warning("Upload an Video OR Untick the Upload Button)")
        st.stop()





