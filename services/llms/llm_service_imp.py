from pydantic import SecretStr
from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.redis import RedisStorage
from typing import Optional


class LLMServiceImp:
    """Create a singleton class for LLMServiceImp that initializes the agent with a knowledge base."""
    _instance = None
    _agent: Optional[Agent] = None
    _secret_key: Optional[SecretStr] = None
    model = "gemini-2.5-flash"
    
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    async def initialize_agent(self, key: str, knowledge_base, user_id: str) -> None:
        """Initializes the agent with the provided knowledge base."""
        self._secret_key = SecretStr(key)
        try:
            memory = Memory(
                db=RedisMemoryDb(
                    prefix="agno_test",
                    host="localhost",
                    port=6379,
                    db=0,
                ),
                model=Gemini(id=self.model)
            )
            storage = RedisStorage(
                prefix="agno_test",
                host="localhost",
                port=6379,
                db=0,
            )
            self._agent = Agent(
                user_id=user_id,
                model=Gemini(id=self.model),
                markdown=True,
                description="You are the 'Oracle' of our company; your goal is to help the employees.",
                instructions=[
                    "Answer the following question in four sentences maximum.",
                    "If you don't know the answer, say 'I don't know. Try the company sector responsable.'"
                ],
                storage=storage,
                memory=memory,
                enable_agentic_memory=True,
                add_history_to_messages=True,
                search_knowledge=True,
                show_tool_calls=True,
                knowledge=knowledge_base,
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize agent: {e}")
    
    async def get_answer(self, query: str) -> None:
        """Processes the query and returns the response from the agent."""
        if self._agent is None:
            raise ValueError("Agent has not been initialized.")
        
        try:
            self._agent.print_response(
                query,
                stream=True,
                show_full_reasoning=True,
                show_tool_calls=True
            )
        except Exception as e:
            raise ValueError(f"Failed to get response from agent: {e}")
