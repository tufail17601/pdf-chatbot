# 📘 PDF Chatbot in Simple Python
# This app lets you upload a PDF, processes it, and allows you to ask questions about it.

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


# -----------------------------------------------------------
# 🧠 STEP 1: Function to read text from uploaded PDFs
# -----------------------------------------------------------
def read_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


# -----------------------------------------------------------
# ✂️ STEP 2: Split text into smaller parts (chunks)
# -----------------------------------------------------------
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


# -----------------------------------------------------------
# 💾 STEP 3: Convert text chunks into vector embeddings
# -----------------------------------------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# -----------------------------------------------------------
# 🤖 STEP 4: Create a chat model that uses the stored knowledge
# -----------------------------------------------------------
def create_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain


# -----------------------------------------------------------
# 💬 STEP 5: Handle user questions and show chat
# -----------------------------------------------------------
def handle_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown("## 👤 You")
            st.write(message.content)
            st.markdown("---")
        else:
            st.markdown("## 🤖 Bot")
            st.write(message.content)
            st.markdown("---")


# -----------------------------------------------------------
# 🚀 STEP 6: Streamlit main app
# -----------------------------------------------------------
def main():
    # Load your API key
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    st.set_page_config(page_title="Chat with your PDF", page_icon="📘")
    st.title("💬 Chat with your PDF")

    # Initialize session variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    # Sidebar for uploading PDFs
    with st.sidebar:
        st.header("📂 Upload your PDF")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("Process PDF"):
            if not pdf_files:
                st.warning("⚠️ Please upload at least one PDF before processing.")
            else:
                with st.spinner("⏳ Processing your PDF..."):
                    try:
                        # Extract text
                        text = read_pdf_text(pdf_files)

                        # Split into chunks
                        chunks = split_text_into_chunks(text)

                        # Create vector store (embeddings)
                        vector_store = create_vector_store(chunks)

                        # Create chat conversation chain
                        st.session_state.conversation = create_conversation_chain(vector_store)

                        st.success("✅ PDF processed! You can now ask questions.")
                    except Exception as e:
                        st.error("❌ Oops! Something went wrong while processing your PDF. Please try again.")

    # Input for user question
    user_question = st.text_input("Ask a question about your PDF:")

    if st.button("Send") and user_question and st.session_state.conversation:
        handle_question(user_question)


if __name__ == "__main__":
    main()
