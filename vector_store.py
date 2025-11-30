from collections import defaultdict
from typing import IO, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database import Base, SourceItem, SourceType


class VectorStoreHelper:
    def __init__(self, gemini_api_key, llama_idx_key, model: BaseChatModel):
        self.parser = LlamaParse(
            api_key=llama_idx_key,
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="text-embedding-004",
            google_api_key=gemini_api_key,
        )
        self.vector_store = Chroma(
            collection_name="testing",
            embedding_function=self.embeddings,
            persist_directory="./chroma_langchain_db",
        )
        self.model = model

        engine = create_engine("sqlite:///sources.sqlite")
        Base.metadata.create_all(engine)
        self.db_session = sessionmaker(bind=engine)

    def add_files(self, files: Sequence[IO[bytes]]):
        fnames = []
        source_items = {}

        with self.db_session() as session:
            for i in files:
                path = f"uploads/{i.name}"
                with open(path, "wb") as file:
                    file.write(i.read())

                fnames.append(path)

                source_item = SourceItem(name=i.name, path=path, type=SourceType.FILE)
                session.add(source_item)
                session.commit()
                source_items[i.name] = source_item.id

        # Just loading data with self.parser does not add any metadata :(
        parsed_docs = SimpleDirectoryReader(
            input_files=fnames, file_extractor={"*": self.parser}
        ).load_data()

        langchain_docs: list[Document] = [
            Document(page_content=d.text, metadata=d.metadata) for d in parsed_docs
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(langchain_docs)

        counters = defaultdict(int)
        for d in all_splits:
            for d in all_splits:
                src = d.metadata.get("file_name") or "unknown"
                page = d.metadata.get("page_label") or "unknown"
                idx = counters[src]
                counters[src] += 1

                d.metadata["source"] = src
                d.metadata["page"] = page
                d.metadata["chunk"] = idx

                d.metadata["src_id"] = source_items[src] or -1

                if "start_index" in d.metadata:
                    start = d.metadata.get("start_index")
                else:
                    start = d.metadata.get("start")

                d.metadata["start"] = start
                d.metadata["end"] = (
                    start + len(d.page_content) if start is not None else None
                )

        self.vector_store.add_documents(all_splits)

    def add_urls(self, urls: list[str]):
        loader = WebBaseLoader(urls)

        docs = loader.load()
        source_items = {}
        with self.db_session() as session:
            for i in docs:
                i.page_content = str(
                    self.model.invoke(
                        """
                You are a specialized webpage parser.  
                You will receive raw webpage content that may include navigation menus,
                ads, scripts, styling, buttons, or repeated elements.

                Your job is to extract ONLY the meaningful human-written content of the page
                and return it as clean, well-structured Markdown.

                Instructions:
                - Remove all navigation items, headers, footers, ads, cookie popups, and scripts.
                - Remove duplicate sections or repeated boilerplate.
                - Keep ONLY the main article/content/important text.
                - Preserve headings, subheadings, lists, tables, and code blocks.
                - Fix broken or split sentences when possible.
                - Do not invent new content — only reorganize what exists.
                - Use concise, clean Markdown.

                Respond ONLY with the cleaned Markdown.\n\n
                Here is the page content:\n
                """
                        + i.page_content
                    ).content
                )
                
                source_item = SourceItem(
                    title=i.metadata["title"],
                    path=i.metadata["source"],
                    type=SourceType.WEBPAGE,
                )
                session.add(source_item)
                session.commit()
                source_items[i.metadata["source"]] = source_item.id

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        counters = defaultdict(int)
        for d in all_splits:
            src = d.metadata.get("source") or "unknown"
            title = d.metadata.get("title") or src
            idx = counters[src]
            counters[src] += 1

            # normalize fields
            d.metadata["source"] = src
            d.metadata["title"] = title
            d.metadata["src_id"] = source_items[src] or -1
            d.metadata["chunk"] = idx

            if "start_index" in d.metadata:
                start = d.metadata.get("start_index")
            else:
                start = d.metadata.get("start")

            d.metadata["start"] = start
            d.metadata["end"] = (
                start + len(d.page_content) if start is not None else None
            )

        self.vector_store.add_documents(all_splits)
        return docs

    def similarity_search(self, query: str, k=4):
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)
