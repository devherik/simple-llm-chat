from abc import ABC, abstractmethod
from langchain_core.documents.base import Document
from typing import List

class LLMService(ABC):
    '''Abstract base class for LLM services.'''
    
    @abstractmethod
    async def __new__(cls, *args, **kwargs):
        pass