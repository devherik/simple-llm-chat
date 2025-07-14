from abc import ABC, abstractmethod

class RAGService(ABC):

    @abstractmethod
    async def _load_data(self) -> None:
        """Load data from the source."""
        pass
    
    @abstractmethod
    async def _process_data(self) -> None:
        """Process the loaded documents."""
        pass