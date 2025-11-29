import os

import streamlit as st
import validators
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent import Agent


def validate_url(url: str):
    """Validates url, does not need url to have https:// as the begininng"""
    try:
        validators.url(url)
        return True
    except validators.ValidationError as _:
        pass

    try:
        validators.url(f"https://{url}")
        return True
    except validators.ValidationError as _:
        pass

    return False


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

left, center, right = st.columns(3)

with right:
    st.title("Sources")
    # List sources here with checkboxes and stuff

    st.subheader("Files")
    if src_files := st.file_uploader(
        "Upload a file here to use it as a source", accept_multiple_files=True
    ):
        agent.vector_store.add_files(src_files)

    st.subheader("URLs")
    if src_url := st.text_input(
        "Enter a URL to a webpage or file here to use it as a source.",
        autocomplete="url",
    ):
        if validate_url(src_url):
            agent.vector_store.add_urls([src_url])
        else:
            st.error("Incorrect URL format.")


with center:
    st.title("Ask Away")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.write("Files uploaded here will not be used as sources for other prompts")
    if user_input := st.chat_input("What can I help with?", accept_file="multiple"):
        files: list[UploadedFile] = user_input["files"]  # type: ignore
        prompt: str = user_input["text"]  # type: ignore

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        retrieved_docs, response = agent.new_prompt(prompt, files)

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
