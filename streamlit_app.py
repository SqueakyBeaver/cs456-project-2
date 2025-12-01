import os

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

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
    print("bad")
    st.stop()


print("Initializing DB connection")
engine = create_engine("sqlite:///app_data.sqlite")
Base.metadata.create_all(engine)
db_session = Session(engine, expire_on_commit=False)
print("DB connection initialized")

print("Initializing agent")
agent = Agent(gemini_api_key, llamaidx_api_key, engine)
print("Agent initialized")

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None
if "chats" not in st.session_state:
    st.session_state.chats = dict()

if not st.session_state.chats and not get_chats(db_session):
    new_chat(db_session, title="First chat")

def update_chats():
    st.session_state.chats = {}
    for chat in get_chats(db_session):
        st.session_state.chats[chat.id] = (chat, chat_page(chat, agent, db_session))


update_chats()
last_chat_id = max([i for i in st.session_state.chats.keys()])

pg = st.navigation(
    [pg for _, pg in st.session_state.chats.values()],
    position="hidden",
)
pg.run()

with st.sidebar:
    with st.container(gap=None, height=400, border=False):
        for chat, page in st.session_state.chats.values():
            with st.container(horizontal=True, vertical_alignment="center", gap=None):
                st.page_link(page)
                if st.button(
                    "", icon=":material/close:", type="tertiary", key=f"close-{chat.id}"
                ):
                    delete_chat(db_session, chat.id)
                    update_chats()
                    st.rerun()

        st.divider()
        if st.button("New chat", width="stretch"):
            new_chat(db_session, title=f"New Chat {last_chat_id + 1}")
            update_chats()
            st.rerun()

