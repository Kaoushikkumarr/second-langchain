# LangChain-2

This project demonstrates Retrieval-Augmented Generation (RAG) pipelines and multi-LLM integration using LangChain, HuggingFace embeddings, FAISS vector store, and Ollama LLMs.  
It also shows how to use OpenAI and GROQ APIs for cloud-based inference.

---

## Features

- **RAG Pipeline**: Ingest data from text, web, and PDF sources, split documents, generate embeddings, store and search vectors, and build retrieval chains for question answering.
- **Local Embeddings**: Use HuggingFace models for free, local embeddings.
- **Vector Store**: Store and search document chunks using FAISS.
- **LLM Integration**: Use Ollama's smallest model (`tinyllama`) for local inference.
- **Cloud LLMs**: Access OpenAI GPT models and GROQ API for cloud inference.
- **Example Notebooks**: Step-by-step workflows in `rag/sample_rag.ipynb` and `rag/chain.ipynb`.

---

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/kaoushikkumarr/langchain-2.git
   cd langchain-2
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. (Optional) Install Ollama locally for LLM inference:
   - Download and install from [Ollama Website](https://ollama.com/download)
   - After installation, run in terminal:
     ```
     ollama run tinyllama
     ```

---

## Usage

### RAG Pipeline Example (see `rag/sample_rag.ipynb`)

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = TextLoader("cc.txt", encoding="utf-8")
text_doc = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(text_doc, embeddings)

query = "problems for Large Industries"
retrieval = db.similarity_search(query=query)
print(retrieval[0].page_content)
```

### Retrieval Chain with Ollama (see `rag/chain.ipynb`)

```python
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant. You will be provided with a context and a question. Use the context to answer the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)
llm = Ollama(model="tinyllama")
doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
db_retriever = db.as_retriever()
retrieval_chain = create_retrieval_chain(
    retriever=db_retriever,
    combine_documents_chain=doc_chain
)

response = retrieval_chain.invoke({"input": query})
print(response["answer"])
```

### Using OpenAI

```python
import openai

openai.api_key = "YOUR_OPENAI_API_KEY"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What is LangChain?"}]
)
print(response.choices[0].message.content)
```

### Using GROQ API

```python
import groq_api

groq_client = groq_api.Client(api_key="YOUR_GROQ_API_KEY")
response = groq_client.generate("What is LangChain?")
print(response)
```

---

## Configuration

Set your API keys as environment variables:
```sh
set OPENAI_API_KEY=your-openai-key
set GROQ_API_KEY=your-groq-key
```

---

## Requirements

- Python 3.11+
- Jupyter Notebook or VS Code
- Ollama (for local LLM inference)

---

## License

MIT

---

## Acknowledgements

- [Ollama](https://ollama.com)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Groq](https://console.groq.com)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Sentence Transformers](https://www.sbert.net/)

---

## Author

- Kaoushik Kumar  
  [GitHub](https://github.com/Kaoushikkumarr)
