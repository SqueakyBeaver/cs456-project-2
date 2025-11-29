from collections import defaultdict
from typing import IO, Sequence

from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse


class VectorStoreHelper:
    def __init__(self, gemini_api_key, llama_idx_key):
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

    def add_files(self, files: Sequence[IO[bytes]]):
        fnames = []
        for i in files:
            path = f"uploads/{i.name}"
            with open(path, "wb") as file:
                file.write(i.read())

            fnames.append(path)

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
            src = d.metadata.get("source") or "unknown"
            idx = counters[src]
            counters[src] += 1

            # normalize fields
            d.metadata["file"] = src
            d.metadata["page"] = "N/A"
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

    def add_urls(self, urls: list[str]):
        import requests
        from bs4 import BeautifulSoup

        docs = []
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")
                text = soup.get_text()
                docs.append(Document(page_content=text, metadata={"source": url}))
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {url}: {e}")

        if not docs:
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(docs)

        counters = defaultdict(int)
        for d in all_splits:
            src = d.metadata.get("source") or "unknown"
            idx = counters[src]
            counters[src] += 1

            # normalize fields
            d.metadata["file"] = src
            d.metadata["page"] = "N/A"
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

    def similarity_search(self, query: str, k=4):
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)
