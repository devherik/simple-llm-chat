import os
from pydantic import SecretStr
from agno.agent import Agent
from agno.models.google import Gemini
from agno.memory.v2.db.redis import RedisMemoryDb
from agno.memory.v2.memory import Memory
from agno.storage.redis import RedisStorage
from agno.tools.telegram import TelegramTools
from typing import Optional, Sequence

from telegram import Audio
from agno.media import Audio as AgnoAudio


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

    async def initialize_agent(self, key: str, knowledge_base) -> None:
        """Initializes the agent with the provided knowledge base."""
        self._secret_key = SecretStr(key)
        try:
            memory = Memory(
                db=RedisMemoryDb(
                    prefix="memory",
                    host="localhost",
                    port=6379,
                    db=0,
                ),
                model=Gemini(id=self.model)
            )
            storage = RedisStorage(
                prefix="agents_sessions",
                host="localhost",
                port=6379,
                db=0,
            )
            self._agent = Agent(
                model=Gemini(id=self.model),
                goal="Your goal is to help the employees.",
                introduction="You are a helpful assistant. Your goal is to help the employees.",
                description="The employee will ask for an assistance about something from the company",
                instructions=[
                    "Provide 'links' to the knowledge base if available.",
                    "Do not provide any information that is not in the knowledge base.",
                    "If you don't know the answer, say something like 'I don't have this information. Try the company sector responsable.'",
                    "Always respond in Portuguese (PT-BR).",
                    "Respond in a friendly and professional tone.",
                    "Use bullet points for lists.",
                    "The answer will be returned to a Telegram bot, so parse accordingly.",
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

    async def get_answer(self, query: str, user_id: str) -> None:
        """Processes the query and returns the response from the agent."""
        if self._agent is None:
            raise ValueError("Agent has not been initialized.")
        
        try:
            response = self._agent.run(query, user_id=user_id)
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to get response from agent: {e}")
        
    
    async def get_answer_by_audio(self, audio: Sequence[Audio], user_id: str) -> None:
        """Processes the audio input and returns the response from the agent."""
        if self._agent is None:
            raise ValueError("Agent has not been initialized.")
        
        try:
            # Convert telegram.Audio to agno.media.Audio
            agno_audios = [
                AgnoAudio(
                    content=audio
                    # Add other fields as required by AgnoAudio
                )
                for a in audio
            ]
            response = self._agent.run(audio=agno_audios, user_id=user_id)
            return response.content
        except Exception as e:
            raise ValueError(f"Failed to get response from agent by audio: {e}")
