# 🤖 RAG Chat Bot with Groq + Streamlit

This is a tiny toy example of a **RAG (Retrieval Augmented Generation)** chatbot built using:
- [LangChain](https://www.langchain.com/)
- [Groq LLM](https://groq.com/)
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Streamlit](https://streamlit.io/) for UI

---

## 🚀 Features
- Upload a `.txt` file
- Ask live questions about the file
- Uses embeddings + FAISS for retrieval
- Answers powered by Groq’s **Llama3-8B-8192**

---

## 🛠️ Installation

Clone this repo and install dependencies:

```bash
pip install langchain langchain-community langchain-groq faiss-cpu sentence-transformers streamlit
