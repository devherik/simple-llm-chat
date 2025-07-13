from abc import ABC, abstractmethod

class RAGService(ABC):

    @abstractmethod
    def retrieve_documents(self, query: str) -> list[str]:
        """Retrieve relevant documents based on the query."""
        pass
    
    @abstractmethod
    def generate_summary(self, documents: list[str]) -> str:
        """Generate a summary from the retrieved documents."""
        pass