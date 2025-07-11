from ..models.llm_model import LLMModel

class LLMService:
    def __init__(self, api_key):
        self.model = LLMModel(api_key)

    def generate_response(self, prompt):
        try:  
            response = self.model.generate_response(prompt)
            if response.status_code != 200:
                raise Exception(f"Error generating response: {response.text}")
            return response.json()
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}