from service.llm_service import LLMService

def main():
    llm_service = LLMService()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break
        response = llm_service.generate_response(user_input)
        print(f"LLM: {response}")

if __name__ == "__main__":
    main()