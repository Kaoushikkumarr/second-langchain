import os

import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv(".env")

# Langchain Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Get API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Please add it to your .env file.")
    st.stop()

# Page title
st.title("Langchain Demo With ChatGroq API")

# Unique key for the text input to avoid duplicate key errors
input_text = st.text_input("Search your query here:", key="query_input_main")

# Define your prompt
prompt = PromptTemplate.from_template("Answer the following question:\n\n{question}")

# Initialize the GROQ model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",  # You can also try: llama2-70b-4096, gemma-7b-it
    temperature=0,
)

# Output parser
output_parser = StrOutputParser()

# Create chain
chain = prompt | llm | output_parser

# Run the chain when the user enters a query
if input_text:
    result = chain.invoke({"question": input_text})
    st.subheader("Answer:")
    st.write(result)
