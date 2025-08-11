import requests
import streamlit as st


def get_openai_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]["content"]


def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]


def get_groq_response(input_text):
    response = requests.post(
        "http://localhost:8000/story/invoke", json={"input": {"topic": input_text}}
    )
    return response.json()["output"]["content"]


# Streamlit Framework
st.title("Langchain Demo with Openai LLAMA2 API Chains")
input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write an poem on")
input_text2 = st.text_input("Write an story on")

if input_text:
    st.write(get_openai_response(input_text))
if input_text1:
    st.write(get_ollama_response(input_text1))
if input_text2:
    st.write(get_groq_response(input_text2))
