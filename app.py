import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Accident Recommendation System")

uploaded_file = st.file_uploader("Upload an accident image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

    st.success("Image uploaded successfully")

    # Add your prediction model here
