# Installing Dependincies
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import pickle

# Global Variable for Image Size
IMG_SIZE = 28



def image_reducer(data, size=IMG_SIZE):
    """
    Arguments:
        path: (str)
    Returns:
        image that is reduced in size: (np.array)
    Function purpose:
        Reduce the input image arrays from size (300, 300) -> (28,28).
    Algorithm:
        1. Reshape input from 3D -> 2D.
        2. Standardize the data by dividing by 255.
        3. Use CV2 to resize the np.array.
        4. Return the reduced image array.
    """

    # 1. Reshape input from 3D -> 2D.
    data = data[:,:, 0]

    # 2. Standardize the data by dividing by 255.
    data = data / 255

    # 3. Use CV2 to resize the np.array.
    reduced_img = cv2.resize(data, (size, size))

    # 4. Return the reduced image array.
    return reduced_img



# Opening the model that was trained in jupyter.
with open("CNN_model_p7.pkl", "rb") as pickle_file:
    CNN_model_p7 = pickle.load(pickle_file)

# The "loaded_model" is used in make prediction function.
# Note: Allthough it does not seem like I am directly calling
# Keras or Tensorflow, in order for this model to be deployed
# on streamlit, I needed to add keras and Tensorflow into the
# requirments.txt file.
loaded_model = CNN_model_p7



def make_prediction(data, IMG_SIZE=IMG_SIZE):
    """
    Arguments:
        data: (np.array)
    Returns:
        predictions: (dict)
    Function purpose:
        This function takes in the image data from the drawn image,
        and makes a prediction whether or not the image is a happy
        or sad face.
    Algorithm:
        1. Reshape input data to be fed into CNN.
        2. Make prediction with loaded model.
        3. Add probabilities from prediction into dictionary.
        4. Return the dictionary.
    """

    # 1. Reshaping the data, into size (1,28,28,1).
    data = data.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # 2. Making prediction with loaded model.
    prediction = loaded_model.predict(data)[0]

    # Add probabilities from prediction into dictionary.
    return_dict = {"Sad": prediction[0], "Happy": prediction[1] }

    # 4. Return the dictionary.
    return return_dict



"""
THIS IS THE BEGGINING OF THE STREAMLIT UI.
"""


import streamlit as st
from streamlit_drawable_canvas import st_canvas

# https://www.youtube.com/watch?v=zhpI6Yhz9_4
# https://pypi.org/project/paul-smiley-prediction/0.0.1/

st.set_page_config(page_title="ML: Smiley App", page_icon = '🙂')



st.write(
"""
# Hand Drawn Smiley Face Prediction with Convolutional Neural Network (Keras Tensor Flow) 🙂
Hello!

In this computer vision project, I collected 1000 hand-drawn smiley faces and sad faces, and using data augmentation generated a dataset of 80,000 images. Then I used these images to train a convolutional neural network that can classify sad vs smiley faces with 99% accuracy.

"""

st.write("[![Star](<https://github.com/fentresspaul61B/Smiley_Predictor_CNN><fentresspaul61B>/<Smiley_Predictor_CNN>.svg?logo=github&style=social)](<https://gitHub.com/><username>/<repo>)")


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
