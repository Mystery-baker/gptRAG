import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import fitz  # PyMuPDF
from transformers import pipeline


def init():
    # Load the OpenAI API key from the environment variable
    load_dotenv()
    
    # Test that the API key exists
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    # Setup Streamlit page
    st.set_page_config(
        page_title="Your own ChatGPT",
        page_icon="🤖"
    )


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text


def load_documents_and_create_vector_store():
    loader = PyPDFLoader("cr7.pdf")  # Replace with your PDF path
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def main():
    init()

    chat = ChatOpenAI(temperature=0)
    retriever = st.session_state.vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat, chain_type="stuff", retriever=retriever)

    sentiment_analyzer = pipeline("sentiment-analysis")

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    st.header("Your own ChatGPT 🤖")

    # Sidebar with user input
    with st.sidebar:
        user_input = st.text_input("Your message: ", key="user_input")

        # Handle user input
        if user_input:
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("Thinking..."):
                response = qa_chain.run(input=user_input)
                sentiment = sentiment_analyzer(response)[0]
                sentiment_message = f"Sentiment: {sentiment['label']} (score: {sentiment['score']:.2f})"
                st.session_state.messages.append(
                    AIMessage(content=f"{response}\n\n{sentiment_message}"))

    # Display message history
    messages = st.session_state.get('messages', [])
    for i, msg in enumerate(messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=str(i) + '_user')
        else:
            message(msg.content, is_user=False, key=str(i) + '_ai')


if __name__ == '__main__':
    # Load documents and create vector store
    st.session_state.vector_store = load_documents_and_create_vector_store()
    main()


