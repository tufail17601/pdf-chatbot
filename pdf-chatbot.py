# 📘 Mobile-Friendly PDF Chatbot

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# -------------------------------
# Step 1: Read PDF text
# -------------------------------
def read_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

# -------------------------------
# Step 2: Split text into chunks
# -------------------------------
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

# -------------------------------
# Step 3: Create vector store
# -------------------------------
def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# -------------------------------
# Step 4: Create conversation chain
# -------------------------------
def create_conversation_chain(vector_store):
    llm = ChatOpenAI(model="gpt-4o-mini")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return chain

# -------------------------------
# Step 5: Handle user question
# -------------------------------
def handle_question(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    # Show only the last user-bot pair (mobile-friendly)
    if st.session_state.chat_history:
        last_pair = st.session_state.chat_history[-2:]  # last user + bot
        for i, message in enumerate(last_pair):
            if i % 2 == 0:
                st.markdown(f"### 👤 You")
                st.write(message.content)
            else:
                st.markdown(f"### 🤖 Bot")
                st.write(message.content)
        st.markdown("---")

# -------------------------------
# Step 6: Streamlit main app
# -------------------------------
def main():
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    st.set_page_config(page_title="Chat with your PDF", page_icon="📘")
    st.title("💬 Chat with your PDF")

    # Initialize session variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar: Upload PDFs
    with st.sidebar:
        st.header("📂 Upload your PDF")
        pdf_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)

        if st.button("Process PDF") and pdf_files:
            with st.spinner("⏳ Processing your PDF..."):
                text = read_pdf_text(pdf_files)
                chunks = split_text_into_chunks(text)
                vector_store = create_vector_store(chunks)
                st.session_state.conversation = create_conversation_chain(vector_store)
            st.success("✅ PDF processed! You can now ask questions.")

    # Input: User question + Send button
    user_question = st.text_input("Ask a question about your PDF:")

    if st.button("Send") and user_question and st.session_state.conversation:
        handle_question(user_question)

if __name__ == "__main__":
    main()
