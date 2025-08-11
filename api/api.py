import os

import streamlit as st
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langserve import add_routes

load_dotenv()


# environment variables call
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Get API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    st.stop()


app = FastAPI(
    title="Langchain Server API with OpenAI",
    version="1.0",
    description="A simple API to demonstrate Langchain Server with different LLMs.",
)

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

openai_model = ChatOpenAI()

# Ollama model
ollama_model = Ollama(model="llama2")

# Groq model
groq_model = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",  # You can also try: llama2-70b-4096, gemma-7b-it
    temperature=0,
)

prompt1 = ChatPromptTemplate.from_template(
    "Write me as essay about {topic} with 100 words."
)
prompt2 = ChatPromptTemplate.from_template(
    "Write me a poem about {topic} for a 5 year old child."
)

add_routes(
    app,
    prompt1 | openai_model,
    path="/essay",
)

add_routes(
    app,
    prompt2 | ollama_model,
    path="/poem",
)

add_routes(
    app,
    prompt2 | groq_model,
    path="/story",
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    print("Server is running on http://localhost:8000")
    print("Available endpoints:")
