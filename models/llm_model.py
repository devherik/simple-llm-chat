import requests

class LLMModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMModel, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key):
        self.api_key = api_key

    def generate_response(self, prompt):
        # Call the Gemini API to generate a response
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": prompt,
            "max_tokens": 100
        }
        response = requests.post(
            "https://gemini.example.com/v1/generate",
            headers=headers,
            json=data
        )
        return response.json()