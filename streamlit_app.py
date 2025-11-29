import os

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import VectorStoreHelper

try:
    assert st.secrets.has_key("GEMINI_API_KEY"), (
        "Missing Gemini API key in .streamlit/secrets.toml"
    )
    assert st.secrets.has_key("UNSTRUCTURED_API_KEY"), (
        "Missing Unstructured API key in .streamlit/secrets.toml"
    )
    gemini_api_key = st.secrets.GEMINI_API_KEY
    llamaidx_api_key = os.environ["LLAMAINDEX_API_KEY"]
except Exception as e:
    print(e)
    exit(1)


def generate_response(input_text):
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", api_key=gemini_api_key)
    st.info(model.invoke(input_text))


st.title("AI Chatbot thingy i have no idea what to call this")

vector_store = VectorStoreHelper(gemini_api_key, llamaidx_api_key)

# if uploaded_files := st.file_uploader(
#     "Add files for context", accept_multiple_files=True
# ):
#     vector_store.add_files(uploaded_files)


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("What can I help with?", accept_file="multiple"):
    if files := user_input["files"]:
        vector_store.add_files(files)  # type: ignore
    
    prompt = user_input["text"]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "assistant", "content": "ur mom"})
