from services.llms.llm_service_imp import LLMService, LLMServiceImp

def main():
    llm_service = LLMServiceImp()
    chat = llm_service.start_chat()
    if chat is None:
        print("Failed to start chat.")
        return
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "thanks", "bye"]:
            print("Exiting chat.")
            break
        prompt = f"Uses three sentences maximum to answer this question: {user_input}"
        response = chat.send_message(
            message = prompt,
            config = {"temperature": 0.5}
        )
        print(f"{llm_service.model}: {response.text}")

if __name__ == "__main__":
    main()