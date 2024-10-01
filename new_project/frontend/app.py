import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Upload a Brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Convert image to bytes
    img_bytes = uploaded_file.getvalue()

    # Send image to FastAPI backend
    response = requests.post("http://localhost:8000/predict/", files={"file": img_bytes})

    if response.status_code == 200:
        prediction = np.array(response.json()['prediction'])
        st.image(prediction[0], caption="Metastasis Segmentation", use_column_width=True)