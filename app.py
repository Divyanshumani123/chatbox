import streamlit as st
import os
import tempfile

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="Samsung Washing Machine RAG Chatbot",
    page_icon="🧺",
    layout="wide"
)

st.title("🧺 Samsung Washing Machine RAG Chatbot")
st.write("Upload a Samsung washing machine manual (HTML) and ask questions.")

# API Key
api_key = st.secrets.get("OPENAI_API_KEY", "")

if not api_key:
    st.warning("Please add OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader(
    "Upload Samsung Manual (.html)",
    type=["html", "htm"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file.write(uploaded_file.read())
        html_path = tmp_file.name

    with st.spinner("Processing document..."):

        loader = UnstructuredHTMLLoader(file_path=html_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        splits = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )

        retriever = vectorstore.as_retriever()

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks.

            Use the following pieces of retrieved context
            to answer the question.

            If you don't know the answer,
            just say that you don't know.

            Keep the answer concise.

            Question: {question}

            Context:
            {context}

            Answer:
            """
        )

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

    st.success("Document processed successfully!")

    query = st.text_input(
        "Ask a question about the washing machine manual:"
    )

    if st.button("Get Answer"):

        if query:

            with st.spinner("Generating answer..."):
                response = rag_chain.invoke(query)

            st.subheader("Answer")
            st.write(response.content)

        else:
            st.warning("Please enter a question.")