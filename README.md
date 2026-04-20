# LangChain Course Projects

A hands-on Python portfolio of **LLM engineering patterns** built with LangChain, OpenAI-compatible models (via OpenRouter), Pinecone, and Tavily.

This repo demonstrates practical AI application skills across:
- prompt engineering and chaining,
- tool-calling agents,
- raw ReAct loop implementation,
- retrieval-augmented generation (RAG),
- observability with LangSmith.

---

## Why This Project Is Recruiter-Relevant

This codebase showcases the ability to move from basic LLM usage to production-minded patterns:
- Building both **framework-assisted** and **from-scratch** agent loops.
- Integrating external tools and APIs for real-world workflows.
- Implementing an end-to-end **RAG pipeline** (ingest -> index -> retrieve -> answer).
- Structuring outputs with schemas for reliable downstream consumption.
- Adding traceability hooks for debugging and monitoring model behavior.

---

## Project Highlights

| File | What It Demonstrates | Key Concepts |
|---|---|---|
| `main_helloworld.py` | Basic LangChain prompt + model chain | `PromptTemplate`, LCEL pipe (`|`), model invocation |
| `main_tavily.py` | Web-search agent with structured output | `create_agent`, `TavilySearch`, Pydantic response schema |
| `main_custom_tool_tavily.py` | Custom tool pattern (drafted example) | `@tool`, custom search wrapper, agent tool integration |
| `1_agent_loop_langchain_tool_calling.py` | Manual agent loop with LangChain tool-calling | `bind_tools`, tool call dispatch, `ToolMessage`, iterative reasoning |
| `3_raw_react_prompt.py` | Raw ReAct agent (no built-in tool-calling abstraction) | prompt protocol design, regex action parsing, stop-token control, scratchpad |
| `ingestion_rag.py` | Document ingestion to vector DB | `TextLoader`, chunking, embeddings, `PineconeVectorStore.from_documents` |
| `main_rag_retrival.py` | Query-time retrieval and answer generation | retriever config, prompt augmentation, LCEL retrieval chain |

---

## Tech Stack

- **Language:** Python 3.13+
- **LLM / Orchestration:** LangChain, LangChain OpenAI
- **Model Access:** OpenRouter-compatible OpenAI client
- **Retrieval:** Pinecone vector store, OpenAI embeddings
- **Search Tooling:** Tavily
- **Observability:** LangSmith (`@traceable`)
- **Utilities:** `python-dotenv`, Pydantic

Dependencies are defined in `pyproject.toml`.

---

## Repository Structure

```text
langchain-course/
├── 1_agent_loop_langchain_tool_calling.py
├── 3_raw_react_prompt.py
├── ingestion_rag.py
├── main_custom_tool_tavily.py
├── main_helloworld.py
├── main_rag_retrival.py
├── main_tavily.py
├── mediumblog1.txt
├── pyproject.toml
└── uv.lock


Setup
  Create a .env file in the project root.
  Add required environment variables (see below).
  Install dependencies with your preferred tool (uv or pip).

Suggested Environment Variables
  OPENROUTER_API_KEY - required for model/embeddings calls
  INDEX_NAME - required for Pinecone ingestion/retrieval scripts
  TAVILY_API_KEY - required for Tavily search scripts
  LANGSMITH_API_KEY (optional but recommended for tracing)
  LANGSMITH_TRACING=true (optional for trace collection)
