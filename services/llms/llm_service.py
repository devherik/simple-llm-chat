from abc import ABC, abstractmethod
from langchain_core.documents.base import Document
from typing import List

class LLMService(ABC):
    
    @abstractmethod
    async def initialize_rag(self):
        pass

    @abstractmethod
    def _embedding_data(self, data: List[Document]) -> None:
        pass
    
    @abstractmethod
    def start_chat(self):
        pass