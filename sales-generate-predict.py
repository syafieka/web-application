import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import pickle


st.write("# Simple Sales Prediction App") 
st.write("This app predicts the **Sales Advertising**!")

st.sidebar.header('User Input Parameters') 

def user_input_features(): #side bar menu
    TV = st.sidebar.slider('Tv', 2.0, 20.0, 5.4) 
    Radio = st.sidebar.slider('Radio', 3.6, 20.0, 6.2)
    Newspaper = st.sidebar.slider('Newspaper', 2.5, 20.0, 13.3)
    data = {'Tv': Tv,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
modelAdvertising = pickle.load(open("modelAdvertising.h5", "rb"))
prediction = modelAdvertising.predict(df)

st.subheader('Prediction')
st.write(prediction)

