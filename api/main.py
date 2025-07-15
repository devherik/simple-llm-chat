from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
from services.llms.llm_service_imp import LLMServiceImp
from services.rag.notion_rag_imp import NotionRAGImp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    await startup_event()
    yield
    # Cleanup can be added here if needed
    # Shutdown event can be added here if needed
    
router = FastAPI(lifespan=lifespan)

async def startup_event():
    print("Starting up the application...")
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

    await llm_service.initialize_agent(
        key=api_key,
        knowledge_base=rag_service.knowledge_base,
        user_id="1"
    )

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/send")
async def send_message(message: str):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return {"Message received": message}