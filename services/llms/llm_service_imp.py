import os
from langchain_core.documents.base import Document
from typing import List
from services.llms.llm_service import LLMService
from services.rag.notion_rag_imp import NotionRAGImp
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from pydantic import SecretStr


class LLMServiceImp(LLMService):
    _instance = None
    _client = None
    _notion_rag = None
    _vector_store = None
    _secret_key = None
    model = "gemini-2.5-flash"

    def __new__(cls, google_api_key: str, *args, **kwargs):
        if not cls._instance:
            cls._google_api_key = google_api_key # Store the key
            cls._secret_key = SecretStr(google_api_key)
            cls._client = ChatGoogleGenerativeAI(
                api_key=cls._secret_key,
                model=cls.model,
                convert_system_message_to_human=True
            )
            cls._instance = super(LLMService, cls).__new__(cls)
            return cls._instance
        cls._google_api_key = google_api_key # Store the key
        cls._secret_key = SecretStr(google_api_key)
        cls._client = ChatGoogleGenerativeAI(
            api_key=cls._secret_key,
            model=cls.model,
            convert_system_message_to_human=True
        )
        return cls._instance
    
    async def initialize_notion_rag(self):
        """Initialize the Notion RAG component asynchronously."""
        if self._notion_rag is None:
            self._notion_rag = await NotionRAGImp.create()
            self._embedding_data(self._notion_rag.docs)
        return self._notion_rag

    def _clean_metadata(self, documents: List[Document]) -> List[Document]:
        """Clean metadata to ensure compatibility with Chroma vector store."""
        cleaned_docs = []
        for doc in documents:
            cleaned_metadata = {}
            if doc.metadata:
                for key, value in doc.metadata.items():
                    # Convert complex objects to strings or skip them
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        cleaned_metadata[key] = value
                    elif isinstance(value, dict):
                        # Flatten dictionary or convert to string
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                                cleaned_metadata[f"{key}_{sub_key}"] = sub_value
                            else:
                                cleaned_metadata[f"{key}_{sub_key}"] = str(sub_value)
                    elif isinstance(value, list):
                        # Convert list to comma-separated string
                        cleaned_metadata[key] = ", ".join(str(item) for item in value)
                    else:
                        # Convert other types to string
                        cleaned_metadata[key] = str(value)
            
            cleaned_docs.append(Document(
                page_content=doc.page_content,
                metadata=cleaned_metadata
            ))
        return cleaned_docs

    def _embedding_data(self, data: List[Document]) -> None:
        try:
            # Clean metadata before embedding
            cleaned_data = self._clean_metadata(data)
            
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self._secret_key) # Pass the stored API key
            self._vector_store = Chroma.from_documents(
                documents=cleaned_data,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
        except Exception as e:
            print(f"An error occurred while embedding data: {e}")
            raise e

    def start_chat(self):
        if self._client is None:
            raise Exception("LLMService client is not initialized")
        if self._vector_store is None:
            raise Exception("Vector store is not initialized. Please embed data first.")
        try:
            self.chat = RetrievalQA.from_chain_type(
                llm=self._client,
                chain_type="stuff",
                retriever=self._vector_store.as_retriever(),
                return_source_documents=True
            )
            return self.chat
        except Exception as e:
            print(f"An error occurred while starting chat: {e}")
            return None
