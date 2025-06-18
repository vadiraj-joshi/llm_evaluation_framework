
from openai import OpenAI
from openai import APIConnectionError, APIStatusError # Import specific exceptions
import time
from src.ports.llm_service import ILLMService

class OpenAIAdapter(ILLMService):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retries = 3
        self.delay = 1 # seconds

    def get_llm_response(self, prompt: str, model_name: str = None) -> str:
        model_to_use = model_name if model_name else self.model
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content.strip()
                return "No response generated."
            except APIConnectionError as e:
                print(f"Connection error to OpenAI API: {e}. Attempt {attempt + 1}/{self.retries}")
                time.sleep(self.delay * (attempt + 1))
            except APIStatusError as e:
                print(f"API status error from OpenAI: {e.status_code} - {e.response}. Attempt {attempt + 1}/{self.retries}")
                time.sleep(self.delay * (attempt + 1))
            except Exception as e:
                print(f"An unexpected error occurred with OpenAI API: {e}. Attempt {attempt + 1}/{self.retries}")
                time.sleep(self.delay * (attempt + 1))
        raise Exception(f"Failed to get LLM response after {self.retries} attempts.")