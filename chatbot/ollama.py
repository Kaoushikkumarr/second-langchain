import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(".env")

# Langchain Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Creating ChatBot
prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistance tool. Please provide response to the user's questions.",
        ),
        ("user", "Question: {question}"),
    ]
)

# Streamlit Framework
st.title("Langchain Demo With Ollama API.")
input_text = st.text_input("Search your query here.")

# Open AI LLM Call
llm = Ollama(
    model="mistral:7b"  # You can also try: mistral:7b-instruct, llama3:8b, etc.
)
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
