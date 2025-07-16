import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
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

# Initialize the Telegram bot application
app = ApplicationBuilder().token(telegram_token).build()

async def start_telegram_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.effective_user:
        print("Update message is None, cannot send reply.")
        return
    await update.message.reply_html(f"Hello {update.effective_user.first_name}! How can I assist you today?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        message = update.message.text
        chat_id = str(update.message.chat_id)
        try:
            response = await LLMServiceImp().get_answer(message, user_id=chat_id)
            if not response:
                await update.message.reply_text("No response from the agent.")
            else:
                await update.message.reply_text(response)
        except Exception as e:
            await update.message.reply_text(f"Error processing message: {str(e)}")
    else:
        print("Update message or text is None, cannot process message.")

app.add_handler(CommandHandler("start", start_telegram_bot))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


# Initialize the FastAPI application

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

    await app.bot.set_webhook(url=f"{api_url}/telegram-webhook/{telegram_token}")

    await llm_service.initialize_agent(
        key=api_key,
        knowledge_base=rag_service.knowledge_base
    )
    
@router.middleware("http")
async def add_process_time_header(request, call_next):
    response = await call_next(request)
    response.headers["X-Process-Time"] = "0ms"
    return response

@router.get("/")
async def root():
    return {"message": "Agno LLM API is running"}

@router.post(f"/telegram-webhook/{telegram_token}")
async def send_message(message: str):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        response = await LLMServiceImp().get_answer(message, user_id="1")
        if not response:
            raise HTTPException(status_code=500, detail="No response from the agent")
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
    