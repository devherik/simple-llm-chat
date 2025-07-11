from services.llm_service import LLMService

def main():
    llm_service = LLMService()
    chat = llm_service.start_chat()
    if chat is None:
        print("Failed to start chat.")
        return
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        response = chat.send_message(user_input)
        print(f"LLM: {response.text}")

if __name__ == "__main__":
    main()