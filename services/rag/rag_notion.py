import os
from typing import List
from services.rag.rag_service import RAGService
from langchain.document_loaders import NotionDBLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

class RagNotionImp(RAGService):
    _instance = None
    docs = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        integration_token = os.getenv("NOTION_INTEGRATION_TOKEN")
        database_id = os.getenv("NOTION_DATABASE_ID")

        if not integration_token:
            raise ValueError("NOTION_INTEGRATION_TOKEN environment variable not set.")
        if not database_id:
            raise ValueError("NOTION_DATABASE_ID environment variable not set.")

        _loader = NotionDBLoader(
            integration_token = integration_token,
            database_id = database_id
        )
        self.docs = _loader.load()
        print(f"Loaded {len(self.docs)} documents from Notion database.")

    def retrieve_vector_store(self) -> Chroma:
        splits = self._splitter()
        if not splits:
            raise Exception("Splits is not initialized")
        self.vector_store = Chroma.from_documents(
            documents = splits,
            persist_directory = "vector_store"  # Ensure this directory exists or is created
        )
        self.vector_store.persist()
        print("Vector store persisted.")
        return self.vector_store

    def retrieve_documents(self, query: str) -> list[str]:
        if self.docs is None:
            raise Exception("RAGService loader is not initialized")
        return [doc.page_content for doc in self.docs if query.lower() in doc.page_content.lower()]

    def generate_summary(self, documents: list[str]) -> str:
        if not documents:
            return "No relevant documents found."
        return " ".join(documents[:3])