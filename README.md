# LangChain-2

This project demonstrates how to integrate multiple language model APIs in Python, including [Ollama.py](https://github.com/ollama/ollama.py), [OpenAI](https://openai.com/), and [GROQ API](https://console.groq.com). Each API can be used independently for text generation and prompt engineering.

## Features

- Query local LLMs using Ollama
- Access OpenAI GPT models for cloud inference
- Use Qroq API for additional language model capabilities
- Example scripts for each API

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/langchain-2.git
   cd langchain-2
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. (Optional) Install Ollama locally:
   - [Ollama Installation Guide](https://github.com/ollama/ollama.py#installation)
   - [GROQ Website for Information]((https://console.groq.com))

## Usage

### Using Ollama

```python
from ollama import Ollama

ollama_client = Ollama()
response = ollama_client.generate("What is LangChain?")
print(response)
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

### Using Qroq API

```python
import qroq_api

qroq_client = qroq_api.Client(api_key="YOUR_QROQ_API_KEY")
response = qroq_client.generate("What is LangChain?")
print(response)
```

## Configuration

Set your API keys as environment variables:
```sh
set OPENAI_API_KEY=your-openai-key
set QROQ_API_KEY=your-qroq-key
```

## License

MIT

## Acknowledgements

- [Ollama](https://github.com/ollama/ollama.py)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Groq](https://console.groq.com)

## Author Name
- Kaoushik Kumar
- [GitHub](https://github.com/Kaoushikkumarr)
