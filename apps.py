import streamlit as st

import numpy as np

import tensorflow as tf
from keras_preprocessing.image import img_to_array
from tensorflow import keras


classifier = keras.models.load_model('pneumoniad.h5')

st.title("Pneumonia Detector")
from PIL import Image

st.sidebar.title("Pneumonia Detector Tool")
uploaded_file=st.sidebar.file_uploader("Choose a X-ray image file")
if uploaded_file is not None:

    if st.sidebar.button("Predict"):
        img=Image.open(uploaded_file)
        img=img.resize((500,400))
        new_img=img.resize((64,64))
        st.image(img)
        new_img=np.dstack([new_img,new_img,new_img])
        new_img=new_img.astype('float32')/255
        x=img_to_array(new_img)
        result=classifier.predict(x[np.newaxis, :])
        if result[0][0]>0.5:
            st.title("Lungs are affected by pneumonia")
            st.subheader("Please go to the Doctor and Take medicine.")

        else:
            st.title("Lungs are normal.No Pneumonia is found.")
