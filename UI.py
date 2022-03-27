# Installing Dependincies
import matplotlib.image as img
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
import pickle
from PIL import Image

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




# THIS IS THE BEGGINING OF THE STREAMLIT UI.



import streamlit as st
from streamlit_drawable_canvas import st_canvas

# https://www.youtube.com/watch?v=zhpI6Yhz9_4
# https://pypi.org/project/paul-smiley-prediction/0.0.1/

st.set_page_config(page_title="ML: Smiley App", page_icon = '🙂')

# Title Image
title_image = Image.open('Smiley_Title.jpeg')
st.image(title_image)

# Project Intro
intro_image = Image.open('Project_Intro.jpeg')
st.image(intro_image)

# Directions
directions_image = Image.open('Directions.jpeg')
st.image(directions_image)

# Happy Label
happy_label = Image.open('Happy_label.jpeg')

# Sad Label
sad_label = Image.open('Sad_label.jpeg')

# Confidence Label
confidence_label = Image.open('confidence.jpeg')

# Code Label
code_label = Image.open('code_label.png')

# Socials label
socials_label = Image.open('connect.jpeg')


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
        # st.write("# Sad")
        st.image(sad_label, width=400)
        st.image(confidence_label)
        st.write("# " + str(sad))
    else:
        # st.write("# Happy")
        st.image(happy_label, width=400)
        st.image(confidence_label)
        st.write( "# " + str(happy))


# st.image(code_label, width=200)
#
#
#
# st.markdown("[![asdfasdf](code_label.png)](https://github.com/fentresspaul61B/Smiley_Predictor_CNN)")

# Link to Github
st.markdown('''
    <a href="https://github.com/fentresspaul61B/Smiley_Predictor_CNN">
        <img src="https://github.com/fentresspaul61B/Kaggle_Files/blob/main/code_label.jpeg?raw=true" width=300/>
    </a>''',
    unsafe_allow_html=True
)

# Link to linkedin
st.markdown('''
    <a href="https://www.linkedin.com/in/paul-fentress-985ab7112/">
        <img src="https://github.com/fentresspaul61B/Kaggle_Files/blob/main/connect.jpeg?raw=true" width=500/>
    </a>''',
    unsafe_allow_html=True
)

# st.write("[![Link to Code](https://img.shields.io/github/stars/fentresspaul61B/Smiley_Predictor_CNN.svg?logo=github&style=social)](https://github.com/fentresspaul61B/Smiley_Predictor_CNN)")

# st.image(socials_label, width=200)
# st.write("[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/paul-fentress-985ab7112/)")
