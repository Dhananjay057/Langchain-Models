from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.header('Research Tool')

user_input = st.text_input("Enter your prompt")

model = ChatGoogleGenerativeAI( model = "gemini-2.0-flash")
result = model.invoke(user_input)

if st.button("Summarize"):
    st.text('result.content')