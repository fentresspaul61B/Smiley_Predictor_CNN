# Installing Dependincies
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np

# Global Variable for Image Size
IMG_SIZE = 28

def image_reducer(data, size=IMG_SIZE):
    """
    Function: image_reducer(path, size=IMG_SIZE)
    Inputs:
    - path
      path is a file path from anywhere,

    - size
      dimension of square image to be reduced into


    Outputs:
    reduced image of data type np.array
    """

    # Reducing from 3d to 2d shape
    data = data[:,:, 0]

    # standardizing data
    data = data / 255

    # Resizeing Image
    reduced_img = cv2.resize(data, (size, size))

    return reduced_img




import pickle

# Uploading the model from pickle
# with open("CNN_Model.pkl", "rb") as pickle_file:
#   CNN_model = pickle.load(pickle_file)

# Trying model trained in python 3.7 enviroment
# CNN_model_p7.pkl
with open("CNN_model_p7.pkl", "rb") as pickle_file:
    CNN_model_p7 = pickle.load(pickle_file)

# import joblib
# loaded_model = joblib.load("CNN_model_joblib.sav")
loaded_model = CNN_model_p7
# opencv-python-headless
# matplotlib==3.5.0
# numpy==1.18.1
# opencv_python==4.5.4.58
# pandas==0.24.2
# streamlit==1.2.0
# streamlit_drawable_canvas==0.8.0
# tensorflow==2.3.1
# Keras==2.4.3
# joblib==0.14.1


def make_prediction(data, IMG_SIZE=IMG_SIZE):
    """
    This Function takes in a path of an image, and resizes
    it to the specified IMG size

    Then reshapes the image into a convolutional input value of
    (1,28,28,1)

    Then calls the model.predict function on this input

    The output is dictionary with two keys: Happy / Sad
    and there respective probabilities (confidence) for
    the prediction

    """
    data = data.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = loaded_model.predict(data)[0]
    return_dict = {"Sad": prediction[0], "Happy": prediction[1] }
    return return_dict





# import My_Smiley_Helper
# import My_Smiley_Model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# https://www.youtube.com/watch?v=zhpI6Yhz9_4
# https://pypi.org/project/paul-smiley-prediction/0.0.1/

st.set_page_config(page_title="ML: Smiley App", page_icon = '🙂')


st.write(
"""
# Model
Draw a happy or sad smiley face and click the button to see the model
prediction.
"""
)

# Drawing Canvas
canvas_result = st_canvas(
    stroke_width=10,
    background_color="White",
    width = 300,
    height= 300,
)

# Flow:
# Click Button --> Collect image data -->
# Reduce Image --> Make Prediction -->
# Print results on screen
if st.button("Predict"):
    data = image_reducer(canvas_result.image_data.astype('float32'))
    # data = My_Smiley_Helper.image_reducer(canvas_result.image_data.astype('float32'))
    predictions = make_prediction(data)
    # predictions = My_Smiley_Model.make_prediction(data)
    sad = predictions["Sad"] # Numerical Confidence Value for Sad
    happy = predictions["Happy"] # Numerical Confidence Value for Happy
    if sad > happy:
        st.write("Sad")
        st.write(sad)
    else:
        st.write("Happy")
        st.write(happy)
