from dotenv import load_dotenv
from services.llms.llm_service_imp import LLMServiceImp
from services.rag.rag_notion import RagNotionImp

def main():
    load_dotenv()
    llm_service = LLMServiceImp()
    rag_service = RagNotionImp()
    query = input("Enter your query: ")
    documents = rag_service.retrieve_documents(query)
    if not documents:
        print("No relevant documents found.")
        return

    summary = rag_service.generate_summary(documents)
    print(f"Summary of relevant documents: {summary}")

    """chat = llm_service.start_chat()
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
        print(f"{llm_service.model}: {response.text}")"""

if __name__ == "__main__":
    main()