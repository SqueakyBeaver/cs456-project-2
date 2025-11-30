import base64
import mimetypes
from typing import IO, Sequence

from langchain.agents.middleware import (
    AgentState,
)
from langchain_core.documents import Document
from langchain_core.messages import FileContentBlock, HumanMessage, TextContentBlock
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store import VectorStoreHelper


def get_mime_type_from_filename(filename):
    # mimetypes.guess_type returns a tuple: (type, encoding)
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type if mime_type else "application/octet-stream"  # Default fallback


class State(AgentState):
    context: list[Document]


class Agent:
    def __init__(self, gemini_api_key, llamaidx_api_key):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", api_key=gemini_api_key
        )
        self.vector_store = VectorStoreHelper(
            gemini_api_key, llamaidx_api_key, self.model
        )

    def new_prompt(self, text: str, files: Sequence[IO[bytes]]):
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
            "If the user attached any files, additionally use those files to help answer the query, and prioritize them\n"
            f"User's query:\n {text}"
        )

        file_blocks: list[FileContentBlock] = []
        for i in files:
            i.seek(0)
            raw = i.read()
            if not raw:
                print("Empty file")
                continue

            b64 = base64.b64encode(raw).decode()
            file_blocks.append(
                FileContentBlock(
                    type="file",
                    base64=b64,
                    mime_type=get_mime_type_from_filename(i.name),
                )
            )

        return retrieved_docs, self.model.stream(
            [
                HumanMessage(
                    content_blocks=[  # type: ignore
                        TextContentBlock(type="text", text=augmented_message_content),
                    ]
                    + file_blocks
                )
            ]
        )
