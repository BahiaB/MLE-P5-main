import streamlit as st
import json
import requests


st.title("Prédiction de tags StackOverflow")

#st.write("## This is a H2 Title!")
x = st.text_input("Please enter your text here", "text goes here")

if st.button("Search tags"):
    data = {'text': x}
    response = requests.post('http://127.0.0.1:8002/predict/', json=data)
    prediction = response.json()['prediction'][0]
    st.write(f"Your suggested Tag are {prediction}")