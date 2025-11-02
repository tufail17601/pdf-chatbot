# 📘 PDF Chatbot in Simple Python
# This app lets you upload a PDF, processes it, and allows you to ask questions about it.

import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

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
    splitter = CharacterTextSplitter(
        separator="\n",      # split by new lines
        chunk_size=1000,     # each chunk has 1000 characters
        chunk_overlap=200    # overlap helps to not lose context
    )
    chunks = splitter.split_text(text)
    return chunks


# -----------------------------------------------------------
# 💾 STEP 3: Convert text chunks into vector embeddings
# -----------------------------------------------------------
def create_vector_store(chunks):
    # Use a FREE model from HuggingFace to create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Store them using FAISS (a fast similarity search library)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store


# -----------------------------------------------------------
# 🤖 STEP 4: Create a chat model that uses the stored knowledge
# -----------------------------------------------------------
def create_conversation_chain(vector_store):
    # Use OpenAI model for answering (you can also use Llama or other)
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Store chat memory (so bot remembers your previous questions)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Build the retrieval-based chatbot chain
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
            st.markdown(f"### 👤 You")
            st.write(message.content)
            st.markdown("---")
    else:
        # Bot message
        st.markdown(f"### 🤖 Bot")
        st.write(message.content)
        st.markdown("---")

# -----------------------------------------------------------
# 🚀 STEP 6: Streamlit main app
# -----------------------------------------------------------
def main():
    # Set your API key here
    from dotenv import load_dotenv
    load_dotenv()

    #modell = ChatOpenAI(model='gpt-4o-mini')

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
            with st.spinner("⏳ Processing your PDF..."):
                # Extract text
                text = read_pdf_text(pdf_files)

                # Split into chunks
                chunks = split_text_into_chunks(text)

                # Create vector store (embeddings)
                vector_store = create_vector_store(chunks)

                # Create chat conversation chain
                st.session_state.conversation = create_conversation_chain(vector_store)

            st.success("✅ PDF processed! You can now ask questions.")

    # Input for user question
    user_question = st.text_input("Ask a question about your PDF:")
    if user_question and st.session_state.conversation:
        handle_question(user_question)


if __name__ == "__main__":
    main()
