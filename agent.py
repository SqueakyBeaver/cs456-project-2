from typing import IO, Sequence

from langchain.agents.middleware import (
    AgentState,
)
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import VectorStoreHelper


class State(AgentState):
    context: list[Document]


class Agent:
    def __init__(self, gemini_api_key, llamaidx_api_key):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", api_key=gemini_api_key
        )
        self.vector_store = VectorStoreHelper(gemini_api_key, llamaidx_api_key)

    def new_prompt(self, text: str, files: Sequence[IO[bytes]], urls: list[str]):
        if files:
            self.vector_store.add_files(files)
        if urls:
            self.vector_store.add_urls(urls)

        retrieved_docs = self.vector_store.similarity_search(text, k=2)
        # Build a docs content block that includes a short source header for
        # each retrieved chunk so the model can cite sources.
        docs_content_parts = []
        for doc, score in retrieved_docs:
            src = doc.metadata.get("file") or "unknown file"
            page = doc.metadata.get("page") or "unknown page"
            header = f"Relevance: {score * 100}%\nSource: `{src} (page {page})`"

            docs_content_parts.append(f"{header}\n{doc.page_content}")

        docs_content = "\n\n".join(docs_content_parts)
        augmented_message_content = (
            "You are a helpful assistant. If you use any of the following context,"
            "be sure to cite the source. Ignore any sources that are not useful."
            "If you cannot find any helpful sources, you may pull from your own knowledge, but warn the user.\n"
            "Use the following sources to answer the user's query:\n"
            f"{docs_content}\n\n"
            f"User's query:\n {text}"
        )

        return retrieved_docs, self.model.stream(augmented_message_content)
