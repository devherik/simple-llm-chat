from abc import ABC, abstractmethod

class LLMService(ABC):

    @abstractmethod
    def generate_response(self, prompt: str) -> str | None:
        pass
    
    @abstractmethod
    def start_chat(self):
        pass