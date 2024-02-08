import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import pickle


st.write("# Simple Sales Prediction App") 
st.write("This app predicts the **Sales Advertising**!")

st.sidebar.header('User Input Parameters') 

def user_input_features(): #side bar menu
    TV = st.sidebar.slider('Tv', 0.0, 300.0, 246.7) 
    Radio = st.sidebar.slider('Radio', 0.0, 50.0, 42.0)
    Newspaper = st.sidebar.slider('Newspaper', 0.0, 200.0, 182.8)
    data = {'TV': TV,
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

