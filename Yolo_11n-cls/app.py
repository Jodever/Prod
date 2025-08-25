import streamlit as st
import image_processor
import yolo11_inference

st.title("Detect Naira")

with st.container(border = True):

    inference = ""

    uploaded_image = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    if st.button("Submit"):
        preprocess = image_processor.image_resize(uploaded_image)

        inference = yolo11_inference.prediction(preprocess)


        @st.dialog("Result")
        def resultOut():
            st.text(inference)

        resultOut()