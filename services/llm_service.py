from google import genai

class LLMService:
    _instance = None
    _client = None
    _model = "gemini-2.5-flash"

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LLMService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self._client = genai.Client()

    def generate_response(self, prompt) -> str | None:
        if self._client is None:
            raise Exception("LLMService client is not initialized")
        try:
            response = self._client.models.generate_content(
                model = self._model,
                contents = prompt,
            )
            return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return f"error {str(e)}"

    def start_chat(self):
        if self._client is None:
            raise Exception("LLMService client is not initialized")
        try:
            self.chat = self._client.chats.create(
                model=self._model,
            )
            return self.chat
        except Exception as e:
            print(f"An error occurred while starting chat: {e}")
            return None