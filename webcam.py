#!/usr/bin/env python
# coding: utf-8

# In[1]:


import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
def app_object_detection():
    
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class webcam(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            classes = open('obj.names').read().strip().split('\n')
            np.random.seed(42)
            colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

            # Give the configuration and weight files for the model and load the network.
            net = cv.dnn.readNetFromDarknet('./data/yolov3.cfg', './data/yolov3_final.weights')
            net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            # net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

            # determine the output layer
            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            # construct a blob from the image
            blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            # r = blob[0, 0, :, :]


            net.setInput(blob)
            # t0 = time.time()
            outputs = net.forward(ln)


            boxes = []
            confidences = []
            classIDs = []
            h, w = img.shape[:2]

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > 0.5:
                        box = detection[:4] * np.array([w, h, w, h])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        box = [x, y, int(width), int(height)]
                        boxes.append(box)
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if len(indices) > 0:
                for i in indices.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in colors[classIDs[i]]]
                    cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                    cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            return img


    st.title("Real Time  Detection Application")

    st.header("Webcam Live Feed")
    st.write("Click on start to use webcam and detect state of safety")
    webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=webcam)
