import os

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from agent import Agent

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

print("Initializing agent")
agent = Agent(gemini_api_key, llamaidx_api_key)
print("Agent initialized")

st.title("AI Chatbot thingy i have no idea what to call this")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.container():
    if user_input := st.chat_input("What can I help with?", accept_file="multiple"):
        files: list[UploadedFile] = user_input["files"]  # type: ignore
        prompt: str = user_input["text"]  # type: ignore

        urls = []
        if st.session_state.urls_input:
            urls = [
                url.strip()
                for url in st.session_state.urls_input.strip().split("\n")
                if url.strip()
            ]

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            if urls:
                st.markdown("**URL Sources:**")
                # Using markdown to "highlight" the URLs
                for url in urls:
                    st.markdown(f"- `{url}`")

        retrieved_docs, response = agent.new_prompt(prompt, files, urls)

        with st.chat_message("assistant"):
            full_response = st.write_stream(response)
            # Show retrieved chunks and their sources below the assistant response
            if retrieved_docs:
                st.markdown("**Retrieved sources:**")
                for d, score in retrieved_docs:
                    src = (
                        d.metadata.get("file")
                        or d.metadata.get("source")
                        or "unknown file (probably a db error)"
                    )
                    page = d.metadata.get("page") or "unknown page"
                    header = f"- `{src} (page {page})` ({score:.2f}% relevant)"

                    st.markdown(header)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
    st.text_area(
        label="Websites to use as sources (one per line)",
        height=100,
        key="urls_input",
        placeholder="Enter one URL per line...",
    )
