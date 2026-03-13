import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("model/malaria_model.h5")

st.title("Malaria Detection using Deep Learning")

st.write("Upload a blood cell image to detect malaria")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image")

    img = img.resize((64,64))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("Parasitized (Malaria Detected)")
    else:
        st.success("Uninfected (Healthy Cell)")