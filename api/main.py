import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Request
from services.llms.llm_service_imp import LLMServiceImp
from services.rag.notion_rag_imp import NotionRAGImp
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    print(f"Warning: .env file not found at {env_path}")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
if not telegram_token or not telegram_chat_id:
    raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables must be set")
api_url = os.getenv("API_URL")
if not api_url:
    raise ValueError("API_URL environment variable must be set")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    await startup_event(app)
    yield
    # Cleanup can be added here if needed
    # Shutdown event can be added here if needed
    
router = FastAPI(lifespan=lifespan)

# Application startup logic
async def startup_event(app: FastAPI):
    print("---> Starting up the application...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    # Initialize services
    rag_service = NotionRAGImp()
    llm_service = LLMServiceImp()
    if llm_service is None or rag_service.knowledge_base is None:
        raise RuntimeError("Failed to initialize services or knowledge base.")

    await llm_service.initialize_agent(
        key=api_key,
        knowledge_base=rag_service.knowledge_base
    )

    # Initialize and configure the Telegram bot application
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable must be set")
    ptb_app = ApplicationBuilder().token(telegram_token).build()
    
    # Store services and the ptb_app instance in bot_data and app.state
    ptb_app.bot_data['llm_service'] = llm_service
    app.state.ptb_app = ptb_app

    # Set the webhook
    webhook_url = f"{api_url}/telegram-webhook/{telegram_token}"
    await ptb_app.bot.set_webhook(url=webhook_url)
    print(f"---> NGROK URL set to: {api_key}")

@router.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Process-Time"] = "0ms"
    return response

@router.get("/")
async def root():
    return {"message": "Agno LLM API is running"}

@router.post(f"/telegram-webhook/{telegram_token}")
async def send_message(request: Request):
    try:
        ptb_app = request.app.state.ptb_app
        update_json = await request.json()
        update = Update.de_json(update_json, ptb_app.bot)
        # await ptb_app.process_update(update)
        if update.message and update.message.text:
            message = update.message.text
            chat_id = str(update.message.chat_id)
            llm_service: LLMServiceImp = ptb_app.bot_data['llm_service']
            response = await llm_service.get_answer(query=message, user_id=chat_id)
            if not response:
                await update.message.reply_text("No response from the agent.")
            else:
                await update.message.reply_text(response)
        else:
            print("Update message or text is None, cannot process message.")
        return {"status": "success", "message": "Message processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
