import streamlit as st
import json
import requests

st.title("Prédiction de tags StackOverflow")

#st.write("## This is a H2 Title!")
x = st.text_input("What is your questionn? ", " Question")

if st.button("Click Me"):
    st.write(f"Your favorite movie is `{x}`")