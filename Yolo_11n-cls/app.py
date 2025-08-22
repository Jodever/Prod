import streamlit as st
import image_processor
import yolo11_inference

st.title("Detect Naira")

st.container()
col1, col2 = st.columns(2, gap="medium")

inference = ""
with col1:
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        preprocess = image_processor.image_resize(uploaded_image)

        inference = yolo11_inference.prediction(preprocess)


with col2:
    st.header("Result")
    st.text(inference)

