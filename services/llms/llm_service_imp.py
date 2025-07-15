from services.llms.llm_service import LLMService
from pydantic import SecretStr
from langchain_core.documents.base import Document
from typing import List
from agno.agent import Agent, AgentKnowledge
from agno.models.google import Gemini
from agno.memory.v2.memory import Memory
from typing import Optional


class LLMServiceImp:
    """Create a singleton class for LLMServiceImp that initializes the agent with a knowledge base."""
    _instance = None
    _agent: Optional[Agent] = None
    _memory: Optional[Memory] = None
    _secret_key: Optional[SecretStr] = None
    _knowledge_base: Optional[AgentKnowledge] = None
    model = "gemini-2.5-flash"
    
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    async def initialize_agent(self, key: str) -> None:
        """Initializes the agent with the provided knowledge base."""
        self._secret_key = SecretStr(key)
        # self._knowledge_base = knowledge_base
        self._agent = Agent(
            model=Gemini(id=self.model),
            markdown=True,
            description="You are the 'Oracle' of our company; your goal is to help the employees.",
            instructions=[
                "Answer the following question in four sentences maximum.",
                "If you don't know the answer, say 'I don't know. Try the company sector responsable.'"
            ],
            # search_knowledge=True,
            # knowledge=self._knowledge_base,
        )
    
    async def get_answer(self, query: str) -> None:
        """Processes the query and returns the response from the agent."""
        if self._agent is None:
            raise ValueError("Agent has not been initialized.")
        
        self._agent.print_response(query, stream=True)
