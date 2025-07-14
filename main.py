from dotenv import load_dotenv
import os
import asyncio
from services.llms.llm_service_imp import LLMServiceImp

async def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    llm_service = LLMServiceImp(google_api_key=api_key)
    if llm_service is None:
        print("Failed to initialize LLM service.")
        return

    # Initialize the Notion RAG component
    await llm_service.initialize_notion_rag()
    
    chat = llm_service.start_chat()
    if chat is None:
        print("Failed to start chat.")
        return
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "thanks", "bye"]:
            print("Exiting chat.")
            break
        print("Thinking...")
        prompt = f"Uses three sentences maximum to answer this question: {user_input}"
        response = chat.run(prompt)
        if response is None:
            print("No response from the chat service.")
            continue
        print(f"{llm_service.model}: {response.text}")

if __name__ == "__main__":
    asyncio.run(main())