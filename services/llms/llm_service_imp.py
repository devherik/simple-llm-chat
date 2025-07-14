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
    model = "gemini-2.5-flash"

    def __new__(cls, google_api_key: str, *args, **kwargs):
        if not cls._instance:
            _secret_key = SecretStr(google_api_key)
            cls._client = ChatGoogleGenerativeAI(
                api_key=_secret_key,
                model=cls.model,
                convert_system_message_to_human=True
            )
            cls._notion_rag = NotionRAGImp()
            cls._instance = super(LLMService, cls).__new__(cls)
            return cls._instance
        _secret_key = SecretStr(google_api_key)
        cls._client = ChatGoogleGenerativeAI(
            api_key=_secret_key,
            model=cls.model,
            convert_system_message_to_human=True
        )
        cls._notion_rag = NotionRAGImp()

    def _embedding_data(self, data: List[Document]) -> None:
        if self._client is None:
            raise Exception("LLMService client is not initialized")
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="model/embedding-001")
            self._vector_store = Chroma.from_documents(
                documents=data,
                embedding=embeddings,
                persist_directory="chroma_db"
            )
        except Exception as e:
            print(f"An error occurred while embedding data: {e}")
            raise e

    def start_chat(self):
        if self._client is None:
            raise Exception("LLMService client is not initialized")
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
