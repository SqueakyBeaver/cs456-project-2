from functools import partial

import streamlit as st
import validators
from streamlit.runtime.uploaded_file_manager import UploadedFile

from agent import Agent
from database import Chat, FileItem, SourceType, db_session, get_sources


def validate_url(url: str):
    """Validates url, does not need url to have https:// as the begininng"""
    try:
        validators.url(url)
    except validators.ValidationError as _:
        pass

    try:
        validators.url(f"https://{url}")
        url = f"https://{url}"
    except validators.ValidationError as _:
        pass

    return url


def _page(chat: Chat, agent: Agent):
    def source_widget(item: FileItem):
        enabled = item in chat.enabled_sources
        with st.container(
            key=f"source_item_{item.id}",
            border=True,
            horizontal_alignment="center",
        ):
            with st.container(
                horizontal=True, gap="medium", horizontal_alignment="distribute"
            ):
                # Checkbox for enabling/disabling the source
                if (
                    st.checkbox(
                        "-",
                        value=enabled,
                        key=f"source-{item.id}-enabled",
                        label_visibility="hidden",
                    )
                    and not enabled
                ):
                    chat.add_enabled_sources([item])
                elif enabled:
                    chat.remove_enabled_sources([item])

                st.markdown(f"### {item.title}", width="content")

                if st.button(
                    "",
                    icon=":material/close:",
                    type="tertiary",
                    help="Remove source",
                    key=f"remove_source_{item.id}",
                ):
                    db_session.delete(item)
                    if item in chat.enabled_sources:
                        chat.remove_enabled_sources([item])
                    db_session.commit()
                    st.rerun(scope="fragment")

            # Display the path of the source below the controls
            st.caption(f"{item.path}", width="content")

    @st.fragment()
    @st.dialog(title="Add Sources", width="large")
    def sources_dialog():
        urls = []
        files = []
        for i in get_sources():
            if i.type == SourceType.FILE:
                files.append(i)
            elif i.type == SourceType.WEBPAGE:
                urls.append(i)

        _, left, right, _ = st.columns([0.1, 0.8, 0.8, 0.1], gap="medium")
        with left:
            st.subheader("Files")
            if src_files := st.file_uploader(
                "Upload a file here to use it as a source",
                accept_multiple_files=True,
                key="new_source_file",
            ):
                with st.status("Adding file to sources") as status:
                    # Merge FileItem objects into the session before adding
                    chat.add_enabled_sources(
                        agent.vector_store.add_files(src_files, status)
                    )

                    db_session.merge(chat)
                    db_session.commit()
                    status.update(label="File added as a source")

            with st.container(
                height=200, border=False, horizontal_alignment="distribute"
            ):
                for i in files:
                    source_widget(i)

        with right:
            st.subheader("URLs")
            if "should_clear_url_field" not in st.session_state:
                st.session_state.should_clear_url_field = False
            if st.session_state.should_clear_url_field:
                st.session_state.new_source_url = ""

            if src_url := st.text_input(
                "Enter a URL to a webpage or file here to use it as a source.",
                autocomplete="url",
                key="new_source_url",
            ):
                if url := validate_url(src_url):
                    with st.status("Adding URL to sources") as status:
                        st.session_state.should_clear_url_field = True

                        new_sources = agent.vector_store.add_urls([url], status)
                        chat.enabled_sources += new_sources
                        db_session.merge(chat)
                        db_session.commit()

                        status.update(label="URL added as a source")
                        st.rerun(scope="fragment")

                else:
                    st.error("Incorrect URL format.")
            st.space()
            with st.container(height=200, border=False):
                for i in urls:
                    source_widget(i)

    st.title("Ask away")

    with st.container(horizontal=True, horizontal_alignment="distribute"):
        num_sources = len(get_sources())
        num_enabled_sources = len(chat.enabled_sources)
        if st.button(
            f"{num_enabled_sources}/{num_sources} sources enabled. Click to add/manage sources"
        ):
            sources_dialog()

        st.button(f"Summarize {num_enabled_sources}", key="summarize_button")

    st.session_state.messages = chat.messages

    for message in st.session_state.messages:
        with st.chat_message(message.author):
            st.markdown(message.text)

    if user_input := st.chat_input(
        "What can I help with?", accept_file="multiple", key="chat"
    ):
        files: list[UploadedFile] = user_input["files"]  # type: ignore
        prompt: str = user_input["text"]  # type: ignore

        msg = chat.add_message(
            "user",
            text=prompt,
        )

        st.session_state.messages.append(msg)
        with st.chat_message("user"):
            st.markdown(prompt)

        retrieved_docs, response = agent.new_prompt(prompt, files)

        with st.chat_message("assistant"):
            full_response = st.write_stream(response)
            response_text = f"{full_response}\n"
            # Show retrieved chunks and their sources below the assistant response
            if retrieved_docs:
                response_text += "**Retrieved sources:**"
                st.markdown("\n\n**Retrieved sources:**")
                for d, score in retrieved_docs:
                    src = (
                        d.metadata.get("file")
                        or d.metadata.get("source")
                        or "unknown file (probably a db error)"
                    )
                    page = d.metadata.get("page") or "unknown page"
                    header = f"- `{src} (page {page})` ({score:.2f}% relevant)"

                    response_text += header

                    st.markdown(header)

            msg = chat.add_message(
                author="assistant",
                text=response_text,
                source_ids=[i.metadata.get("src_id") for i, _ in retrieved_docs],  # type: ignore
            )

            st.session_state.messages.append(msg)

    if st.session_state.get("summarize_button", None):
        enabled_sources = chat.enabled_sources
        with st.chat_message("user"):
            msg = chat.add_message(
                "user",
                "summarize these files",
                attachment_ids=[i.id for i in enabled_sources],
            )
            st.session_state.messages.append(msg)
            source_titles = [f"`{i.title}`" for i in enabled_sources]
            st.markdown(f"Summarize these sources: {','.join(source_titles)}")

        with st.chat_message("assistant"):
            full_response = st.write_stream(agent.summarize(chat.enabled_sources))
            msg = chat.add_message(
                "assistant",
                str(full_response),
                attachment_ids=[i.id for i in enabled_sources],
            )
            st.session_state.messages.append(msg)


def chat_page(chat: Chat, agent: Agent):
    return st.Page(
        partial(_page, chat=chat, agent=agent),
        title=chat.title,
        url_path=f"chat-{chat.id}",
    )
