import os

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import nest_asyncio
from agent import Agent
from chat import chat_page
from database import Base, delete_chat, get_chats, new_chat

try:
    assert st.secrets.has_key("GEMINI_API_KEY"), (
        "Missing Gemini API key in .streamlit/secrets.toml"
    )
    assert st.secrets.has_key("LLAMAINDEX_API_KEY"), (
        "Missing Llamaparse API key in .streamlit/secrets.toml"
    )
    gemini_api_key = st.secrets.GEMINI_API_KEY
    llamaidx_api_key = os.environ["LLAMAINDEX_API_KEY"]
except Exception as e:
    print(e)
    st.stop()



print("Initializing DB connection")
engine = create_engine("sqlite:///app_data.sqlite")
Base.metadata.create_all(engine)
db_session = Session(engine)
print("DB connection initialized")

print("Initializing agent")
agent = Agent(gemini_api_key, llamaidx_api_key, engine)
print("Agent initialized")

# Get largest chat id from db
st.session_state.chats = get_chats(db_session)
if not st.session_state.chats:
    new_chat(db_session, title="First chat")
    st.session_state.chats = get_chats(db_session)

last_chat_id = max([i.id for i in st.session_state.chats])


@st.fragment
def chat_list():
    with st.container(gap=None, height=400, border=False):
        for chat in st.session_state.chats:
            container = st.container(
                horizontal=True, vertical_alignment="center", gap=None
            )
            container.page_link(chat_page(chat, agent))
            if container.button(
                "", icon=":material/close:", type="tertiary", key=f"{chat}"
            ):
                delete_chat(db_session, chat)
                st.session_state.chats = get_chats(db_session)
                st.rerun(scope="fragment")

        if st.button("New chat", width="stretch"):
            new_chat(db_session, title=f"New Chat {last_chat_id + 1}")

            st.session_state.chats = get_chats(db_session)
            # force a rerun so components are rebuilt with updated chats
            st.rerun(scope="fragment")


with st.sidebar:
    chat_list()
