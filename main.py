from dotenv import load_dotenv
import os
import asyncio
from services.llms.llm_service_imp import LLMServiceImp
from services.rag.notion_rag_imp import NotionRAGImp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

async def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    rag_service = NotionRAGImp()
    llm_service = LLMServiceImp()
    if llm_service is None:
        print("Failed to initialize LLM service.")
        return

    if rag_service.knowledge_base is None:
        print("Failed to initialize knowledge base.")
        return

    await llm_service.initialize_agent(api_key, rag_service.knowledge_base)
    
    input_query = "Como eu cadastro um novo motorista?"

    await llm_service.get_answer(input_query)

if __name__ == "__main__":
    asyncio.run(main())