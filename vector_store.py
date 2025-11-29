from typing import IO, Sequence

from langchain_chroma import Chroma
from langchain_core.documents.base import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

        parsed_docs = self.parser.load_data(fnames)
        langchain_docs = [
            Document(page_content=d.text, metadata=d.metadata) for d in parsed_docs
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        all_splits = text_splitter.split_documents(langchain_docs)

        ids = self.vector_store.add_documents(all_splits)
        return ids

    def similarity_search(self, query: str, k=4):
        return self.vector_store.similarity_search(query, k=k)
