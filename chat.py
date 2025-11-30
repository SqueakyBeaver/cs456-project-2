from functools import partial

import streamlit as st
import validators
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent import Agent
from database import Chat


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


def _page(chat: Chat, agent: Agent):
    @st.dialog("Add Sources", width="large")
    def add_sources():
        left, right = st.columns(2)
        with left:
            st.subheader("Files")
            if src_files := st.file_uploader(
                "Upload a file here to use it as a source", accept_multiple_files=True
            ):
                agent.vector_store.add_files(src_files)

        with right:
            st.subheader("URLs")
            if src_url := st.text_input(
                "Enter a URL to a webpage or file here to use it as a source.",
                autocomplete="url",
            ):
                if validate_url(src_url):
                    x = agent.vector_store.add_urls([src_url])
                    x
                else:
                    st.error("Incorrect URL format.")

    st.title("Ask Away")

    with st.container(horizontal=True):
        num_enabled_sources = 16
        if st.button(
            f"{num_enabled_sources} sources enabled. Click to add/manage sources"
        ):
            add_sources()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input(
        "What can I help with?", accept_file="multiple", key="chat"
    ):
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


def chat_page(chat: Chat, agent: Agent):
    return st.Page(
        partial(_page, chat=chat, agent=agent),
        title=chat.title,
        url_path=f"chat-{chat.id}",
    )
