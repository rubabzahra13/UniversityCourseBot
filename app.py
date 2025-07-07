# app.py
import streamlit as st
from qa_bot import ask_question

st.set_page_config(page_title="University Assistant")
st.title("🎓 University Course Assistant Bot")

query = st.text_input("Ask a question:")

if query:
    answer = ask_question(query)
    st.write("**Answer:**", answer)
