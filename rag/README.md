This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using LangChain, HuggingFace embeddings, FAISS vector store, and Ollama LLMs.  
It shows how to ingest data from text, web, and PDF sources, split documents, generate embeddings, store and search vectors, and build a retrieval chain for question answering.

---

## Features

- **Data Ingestion**: Load documents from text files, web pages, and PDFs.
- **Document Splitting**: Split documents into chunks for embedding.
- **Embeddings**: Use HuggingFace models for local, free embeddings.
- **Vector Store**: Store and search document chunks using FAISS.
- **LLM Integration**: Use Ollama's smallest model (`tinyllama`) for local inference.
- **Retrieval Chain**: Combine retriever and LLM for context-aware Q&A.

---

## Setup

1. **Clone the repository** and navigate to the `rag` folder.
2. **Install dependencies**:
   ```bash
   pip install langchain langchain-community sentence-transformers faiss-cpu python-dotenv beautifulsoup4
   ```
3. **Install Ollama (for local LLM):**
   - Download and install from [https://ollama.com/download](https://ollama.com/download)
   - After installation, run in terminal:
     ```
     ollama run tinyllama
     ```
4. **Add your data files**:
   - Place `cc.txt` and `bb.pdf` in the appropriate directory.
   - Update file paths in the notebook if needed.

---

## Usage

Open and run `sample_rag.ipynb` or `chain.ipynb` in Jupyter Notebook or VS Code.

### Example Workflow

#### 1. Load a text file
```python
from langchain_community.document_loaders import TextLoader
loader = TextLoader("cc.txt", encoding="utf-8")
text_doc = loader.load()
```

#### 2. Load a web page
```python
import bs4
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader(
    web_path="https://kb.objectrocket.com/elasticsearch/how-to-use-python-helpers-to-bulk-load-data-into-an-elasticsearch-index",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("wrap", "content", "sidebar", "table-of-contents")))
)
web_doc = loader.load()
```

#### 3. Load a PDF
```python
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("bb.pdf")
pdf_doc = loader.load()
```

#### 4. Split documents
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
text_split = text_splitter.split_documents(pdf_doc)
```

#### 5. Generate embeddings and store in FAISS
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(text_split, embeddings)
```

#### 6. Semantic search
```python
query = "problems for Large Industries"
retrieval = db.similarity_search(query=query)
print(retrieval[0].page_content)
```

#### 7. Build a retrieval chain with Ollama LLM
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
```

#### 8. Query the retrieval chain
```python
query = "problems for Large Industries"
response = retrieval_chain.invoke({"input": query})
print(response["answer"])
```

---

## Notes

- No API keys are required for HuggingFace embeddings or Ollama.
- The first run will download the HuggingFace model.
- Ollama must be installed and running for LLM inference.
- Make sure your data files are in the correct directory.

---

## Requirements

- Python 3.11+
- Jupyter Notebook or VS Code
- Ollama (for local LLM)

---

## License

MIT