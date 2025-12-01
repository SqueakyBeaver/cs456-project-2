import base64
import mimetypes
from typing import IO, Sequence

from langchain.agents.middleware import (
    AgentState,
)
from langchain_core.documents import Document
from langchain_core.messages import FileContentBlock, HumanMessage, TextContentBlock
from langchain_google_genai import ChatGoogleGenerativeAI
from llama_parse import LlamaParse

from database import FileItem
from vector_store import VectorStoreHelper


class State(AgentState):
    context: list[Document]


class Agent:
    def __init__(self, gemini_api_key, llamaidx_api_key):
        self.model = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", api_key=gemini_api_key
        )
        self.vector_store = VectorStoreHelper(
            gemini_api_key,
            llamaidx_api_key,
            self.model,
        )
        self.file_parser = LlamaParse(api_key=llamaidx_api_key)

    def create_file_block(
        self, file: IO[bytes] | None = None, file_item: FileItem | None = None
    ):
        # mimetypes.guess_type returns a tuple: (type, encoding)
        if file:
            name = file.name
        elif file_item:
            name = file_item.path
        mime_type, _ = mimetypes.guess_type(name)  # type:ignore
        gemini_supported_mimetypes = [
            # Images
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/heic",
            "image/heif",
            # Audio
            "audio/aac",
            "audio/flac",
            "audio/mp3",
            "audio/m4a",
            "audio/mpeg",
            "audio/mpga",
            "audio/mp4",
            "audio/ogg",
            "audio/pcm",
            "audio/wav",
            "audio/webm",
            # Video
            "video/x-flv",
            "video/quicktime",
            "video/mpeg",
            "video/mpegs",
            "video/mpg",
            "video/mp4",
            "video/webm",
            "video/wmv",
            "video/3gpp",
            # Documents/Text
            "application/pdf",
            "text/plain",
        ]

        if file:
            file.seek(0)
            raw_bytes = file.read()
        elif file_item:
            raw_bytes = file_item.raw_bytes

        if mime_type not in gemini_supported_mimetypes:
            docs = self.file_parser.load_data(raw_bytes, extra_info={"file_name": name})

            return [
                FileContentBlock(
                    type="file",
                    base64=base64.b64encode(i.text.encode()).decode(),
                    mime_type="text/plain",
                )
                for i in docs
            ]

        return [
            FileContentBlock(
                type="file",
                base64=base64.b64encode(raw_bytes).decode(),
                mime_type=mime_type,
            )
        ]

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

            file_blocks += self.create_file_block(i)

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

    def summarize(self, files: Sequence[FileItem]):
        file_blocks = []
        for i in files:
            file_blocks += self.create_file_block(file_item=i)

        prompt = """
        You will be given one or more files (PDF, TXT, DOCX, Markdown, or other text-based formats). Your task is to produce a clear, accurate, and concise summary of the combined contents. Follow these rules:
        Read all provided files and treat them as a unified information set.
        Identify the key ideas, major topics, important data points, and recurring themes.
        Do not include unnecessary detail—focus on essential information only.
        Preserve meaning: ensure the summary reflects the original content without introducing new assumptions.
        If the files contain multiple topics, organize the summary with logical sections or bullet points.
        If any file is unreadable or empty, state this clearly but continue summarizing the rest.
        IMPORTANT: If you reference any currency values or dollar amounts, you must escape all dollar signs by prefixing them with a backslash (\\$).
        Example: write \\$1500 instead of $1500.
        Apply this everywhere a dollar sign would appear, including inside code blocks.
        Do not mention file formats unless relevant to content.
        Output format:
        A concise overall summary (1–3 paragraphs).
        Followed by bullet-point highlights of the most important information.
        All dollar signs escaped.
        Begin once the files are provided.
"""

        return self.model.stream(
            [
                HumanMessage(
                    content_blocks=[  # type: ignore
                        TextContentBlock(type="text", text=prompt),
                    ]
                    + file_blocks
                )
            ]
        )
