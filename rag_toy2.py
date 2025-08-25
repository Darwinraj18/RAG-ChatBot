import os
import streamlit as st
from pathlib import Path
import tempfile

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Streamlit UI

st.set_page_config(page_title="RAG Bot", page_icon="🤖")
st.title("🤖 RAG CHAT BOT with Groq")


#  set api key  here
os.environ["GROQ_API_KEY"] = "this_dummy_api_key"


# File uploader
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    # Save file temp
    tempdir = tempfile.mkdtemp()
    path = Path(tempdir) / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split
    loader = TextLoader(str(path))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_documents(docs)

    # Vector DB
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM
    llm = ChatGroq(model="llama3-8b-8192", temperature=0, api_key=os.environ["GROQ_API_KEY"])

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use ONLY the context to answer."),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    # Format docs
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # User query
    query = st.text_input("Ask a question about the uploaded text:")
    if query:
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(query)

        st.markdown("### ✅ Answer:")
        st.write(response)

        with st.expander("🔎 Retrieved Context"):
            for d in retriever.get_relevant_documents(query):
                st.markdown(d.page_content)

else:
    st.info("👆 Upload a `.txt` file to start.")
