# Real-time-Detection-of-PPE-With-Streamlit

## Introduction

> A Computer Vision demo using YOLOv3 on custom Dataset.\
> The user can choose between detection on an image,video or webcam.

## Installation

> Though the project will be deployed for the demo of my project, if you wish to replicate the code the following are the requirements list:
- [GitCLI](https://cli.github.com/ "`GitCLI`")
- [Anaconda](https://www.anaconda.com/) (optional)
- streamlit -> `pip3 install streamlit`
- `git clone https://github.com/Abdelrahmanibrahimrezk/Real-time-Detection-of-PPE-With-Streamlit.git
- `pip install -r requirements.txt`

## Run

Locally:
> Run with command `streamlit run app.py`
> You can add any picture or video and the video detection will be saved in `detections/`

## Code Samples
Image detections:
```python
def detect(img):

    # Load names of classes and get random colors
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
```

Streamlit and detection call:
```python
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
method = st.sidebar.radio('Go To ->', options=['Webcam', 'Image',"Video"])

if method == 'Image':
    image_input()
elif method == "Video":
    video_input()

else:
    app_object_detection()
```

***Abdelrahman Rezk - 2022 &infin;***


