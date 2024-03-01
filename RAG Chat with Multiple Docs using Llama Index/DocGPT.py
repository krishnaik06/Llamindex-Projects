import os, shutil
from dotenv import load_dotenv
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from pypdf import PdfReader
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import google.generativeai as genai


load_dotenv()


def setQueryEngine():
    documents = SimpleDirectoryReader("temp").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    st.session_state.query_engine = index.as_query_engine()


def saveDoc(pdf_docs):
    for uploadedfile in pdf_docs:
        with open(os.path.join("data", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())


def deleteDoc():
    folder = "data"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def index_documents(pdf_docs):
    saveDoc(pdf_docs)
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    deleteDoc()
    return index


def chatBot():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "query_engine" not in st.session_state:
        st.session_state.query_engine = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = st.session_state.query_engine.query(prompt)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )


def main():
    st.set_page_config("Chat Documents")
    st.header("Chat with Multiple Documents(.pdf, .doc, .csv, etc) using LlamaIndex🦙")

    option = st.selectbox(
        "Select which LLM Service You want to use?",
        ("gpt-3.5-turbo", "gemini-pro"),
    )

    if "llm" not in st.session_state:
        st.session_state.llm = OpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=256,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    if st.button("Select"):
        if option == "gemini-pro":
            st.session_state.llm = genai.configure(
                api_key=os.environ.get("GOOGLE_API_KEY"),
                client_options={"api_endpoint": "generativelanguage.googleapis.com"},
            )

    Settings.llm = st.session_state.llm

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if "query_engine" not in st.session_state:
                    setQueryEngine()
                else:
                    indexes = index_documents(pdf_docs)
                    print(indexes)
                    st.session_state.query_engine = indexes.as_query_engine()

    chatBot()


if __name__ == "__main__":
    main()
