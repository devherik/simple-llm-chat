from pydantic import SecretStr
from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.memory import Memory
from typing import Optional


class LLMServiceImp:
    """Create a singleton class for LLMServiceImp that initializes the agent with a knowledge base."""
    _instance = None
    _agent: Optional[Agent] = None
    _memory: Optional[Memory] = None
    _secret_key: Optional[SecretStr] = None
    model = "gemini-2.5-flash"
    
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    async def initialize_agent(self, key: str, knowledge_base) -> None:
        """Initializes the agent with the provided knowledge base."""
        self._secret_key = SecretStr(key)
        self._agent = Agent(
            model=Gemini(id=self.model),
            markdown=True,
            description="You are the 'Oracle' of our company; your goal is to help the employees.",
            instructions=[
                "Answer the following question in four sentences maximum.",
                "If you don't know the answer, say 'I don't know. Try the company sector responsable.'"
            ],
            search_knowledge=True,
            show_tool_calls=True,
            knowledge=knowledge_base,
        )
    
    async def get_answer(self, query: str) -> None:
        """Processes the query and returns the response from the agent."""
        if self._agent is None:
            raise ValueError("Agent has not been initialized.")
        
        try:
            self._agent.print_response(query, stream=True)
        except Exception as e:
            raise ValueError(f"Failed to get response from agent: {e}")
