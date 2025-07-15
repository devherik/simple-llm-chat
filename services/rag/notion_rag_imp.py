import os
from agno.knowledge.document import DocumentKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.base import Document as AgnoDocument
from agno.embedder.google import GeminiEmbedder
from typing import Optional
from langchain.document_loaders import NotionDBLoader
from langchain_core.documents.base import Document

from helpers.cleaner import clean_metadata

class NotionRAGImp:
    """Class to handle data operations for Notion vector storage.
    This class is responsible for loading, and processing data from Notion.
    It also provides methods to retrieve and summarize documents.
    """
    _instance = None
    _vector_db = None
    docs = None
    knowledge_base: Optional[DocumentKnowledgeBase] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self._load_data()
        self._process_data()

    def _load_data(self) -> None:
        # Logic to load data from the Notion database
        token = os.getenv("NOTION_INTEGRATION_TOKEN")
        id = os.getenv("NOTION_DATABASE_ID")
        if not token or not id:
            raise ValueError(
                "Environment variables NOTION_INTEGRATION_TOKEN and NOTION_DATABASE_ID must be set.")
        loader = NotionDBLoader(
            integration_token=token,
            database_id=id,
            request_timeout_sec=30
        )
        self.docs = loader.load()
        print(f"Loaded {len(self.docs)} documents to be used as knowledge.")

    def _process_data(self) -> None:
        # Logic to process the loaded data
        try:
            self._vector_db = ChromaDb(
                collection="knowledge_base",
                path="./chroma_db",
                persistent_client=True,
                embedder=GeminiEmbedder(),
            )
            if self.docs is None:
                raise ValueError("No documents loaded from Notion.")
            # Convert documents to agno.document.base.Document if necessary
            agno_docs = []
            for doc in self.docs:
                if isinstance(doc, AgnoDocument):
                    agno_docs.append(doc)
                elif isinstance(doc, Document):
                    agno_docs.append(
                        AgnoDocument(
                            content=doc.page_content,
                            meta_data=doc.metadata
                        )
                    )
                elif isinstance(doc, dict):
                    agno_docs.append(AgnoDocument(**doc))
                else:
                    raise TypeError("All documents must be convertible to agno.document.base.Document")
            
            cleaned_docs = clean_metadata(agno_docs)
            
            self.knowledge_base = DocumentKnowledgeBase(
                documents=cleaned_docs,
                vector_db=self._vector_db,
                chunking_strategy=FixedSizeChunking(
                    chunk_size=1000,
                    overlap=200,
                )
            )
            self.knowledge_base.load(recreate=False)
        except Exception as e:
            raise ValueError(f"Failed to initialize vector database: {e}")
