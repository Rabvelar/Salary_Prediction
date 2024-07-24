import streamlit as st

print("Before importing predict_page")
from predict_page import show_predict_page
print("After importing predict_page")

print("Before importing explore_page")
from explore_page import show_explore_page
print("After importing explore_page")

page = st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
