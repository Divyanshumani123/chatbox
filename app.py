import os
import tempfile
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Samsung Washing Machine RAG Chatbot",
    page_icon="🧺",
    layout="wide"
)

st.title("🧺 Samsung Washing Machine RAG Chatbot")
st.write("Upload a Samsung Washing Machine HTML manual and ask questions.")

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Instructions")
    st.write("""
1. Upload the Samsung manual (.html)
2. Wait for processing
3. Ask questions about the manual
""")

# -------------------- API KEY --------------------
st.sidebar.title("🔑 OpenAI API Key")

api_key = st.sidebar.text_input(
    "Enter your OpenAI API Key",
    type="password",
    placeholder="sk-..."
)

if not api_key:
    st.info("👈 Please enter your OpenAI API Key in the sidebar to begin.")
    st.stop()


# -------------------- LOAD RAG --------------------
@st.cache_resource(show_spinner=False)
def create_rag_pipeline(html_path):

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

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful Samsung washing machine assistant.

Answer ONLY using the provided context.

If the answer is not available in the context,
reply with:

"I couldn't find that information in the uploaded manual."

Context:
{context}

Question:
{question}

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

    return rag_chain


# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload Samsung Manual (.html)",
    type=["html", "htm"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        tmp.write(uploaded_file.read())
        html_path = tmp.name

    with st.spinner("Processing document..."):

        try:
            rag_chain = create_rag_pipeline(html_path)
            st.success("✅ Document processed successfully!")

        except Exception as e:
            st.error(f"Error while processing document:\n\n{e}")
            st.stop()

    st.divider()

    question = st.text_input(
        "Ask a question:"
    )

    if st.button("Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:

            with st.spinner("Generating answer..."):

                try:
                    response = rag_chain.invoke(question)

                    st.subheader("Answer")
                    st.write(response.content)

                except Exception as e:
                    st.error(f"Error:\n\n{e}")
