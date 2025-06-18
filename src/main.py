import uvicorn
import os
import sys
import yaml # Used here for config check and writing default

# Dynamically add the project root to the path for imports to work
# This is crucial for relative imports like 'from application.app import app'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from application.app import app # Import the FastAPI app instance

if __name__ == "__main__":
    # Ensure configs directory and config.yaml exist
    config_dir = "configs"
    config_file_path = os.path.join(config_dir, "config.yaml")

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)

    if not os.path.exists(config_file_path):
        with open(config_file_path, 'w') as f:
            f.write('openai_api_key: "YOUR_OPENAI_API_KEY_HERE"\n')
        print(f"\n--- IMPORTANT ---")
        print(f"Created a placeholder config file at {config_file_path}.")
        print("Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual OpenAI API key.")
        print("The application cannot start without a valid API key.")
        print(f"--- IMPORTANT ---\n")
        sys.exit(1) # Exit to prompt user to update config

    # Check if the API key is still the placeholder value
    try:
        with open(config_file_path, 'r') as f:
            config_content = yaml.safe_load(f)
            if config_content and config_content.get("openai_api_key") == "YOUR_OPENAI_API_KEY_HERE":
                print(f"\n--- IMPORTANT ---")
                print(f"Your OpenAI API key in {config_file_path} is still the placeholder value.")
                print("Please update it with your actual key before running the application.")
                print("The application cannot start without a valid API key.")
                print(f"--- IMPORTANT ---\n")
                sys.exit(1)
    except Exception as e:
        print(f"Warning: Could not read or parse config file during startup check: {e}")
        # Continue starting, as the app.py will also check it.

    print("Starting LLM Evaluation Framework API...")
    print("Access FastAPI interactive documentation (Swagger UI) at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
    uvicorn.run(app, host="127.0.0.1", port=8000)

