# OpenAI Agent Example

This notebook demonstrates how to build a LangChain agent using OpenAI's function-calling capabilities.  
It shows how to integrate tools, prompts, and OpenAI LLMs to create a conversational agent that can answer questions and perform actions.

---

## Features

- **OpenAI Function-Calling Agent**: Uses `create_openai_functions_agent` for advanced tool use and reasoning.
- **Tool Integration**: Add custom tools (e.g., Wikipedia, Arxiv) for the agent to use.
- **Prompt Engineering**: Use hub-based or custom prompts for agent instructions.
- **Conversational Q&A**: Agent can answer user queries and invoke tools as needed.

---

## Setup

1. **Install dependencies**:
   ```sh
   pip install langchain openai python-dotenv
   ```

2. **Set your OpenAI API key**:
   - Add your key to a `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```
   - Or set it in your environment:
     ```sh
     set OPENAI_API_KEY=your_openai_api_key
     ```

---

## Usage

### 1. Import and configure OpenAI agent

```python
from langchain.agents import create_openai_functions_agent
from langchain_community.llms import OpenAI
from langchain.prompts import ChatPromptTemplate

llm = OpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Use tools and context to answer questions."
)
```

### 2. Add tools

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [wiki_tool]
```

### 3. Create the agent

```python
agent = create_openai_functions_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)
```

### 4. Run the agent

```python
query = "What is LangChain? Search Wikipedia."
response = agent.invoke({"input": query})
print(response["answer"])
```

---

## Notes

- Requires an OpenAI API key and internet access.
- Function-calling agents are only supported with OpenAI models (e.g., GPT-4, GPT-3.5).
- For local LLMs (Ollama), use retrieval/document chains instead.

---

## License

MIT