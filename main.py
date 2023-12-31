import streamlit as st
from PIL import Image
import pickle as pkl
import numpy as np

class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positve'}

# Set the page title
st.title('Sentiment analysis from Vietnamese students’ feedback')

# Load and display an image
image = Image.open('feedback.jpg')
st.image(image, caption='Students’ Feedback Image', use_column_width=True)

# Load the encoder and model
input_ec = open('ec_vsfc.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vsfc.pkl', 'rb')
model = pkl.load(input_md)

# Section for user input
st.header('Write a feedback')
txt = st.text_area('Enter your feedback here:', '')

# Make prediction when the "Predict" button is clicked
if txt != '':
    if st.button('Predict'):
        feature_vector = encoder.transform([txt])
        label = str((model.predict(feature_vector))[0])

        # Display the result
        st.header('Result')
        result_text = f'Sentiment: {class_list[label]}'
        st.markdown(f'<p style="color:#1E90FF;font-size:20px">{result_text}</p>', unsafe_allow_html=True)
