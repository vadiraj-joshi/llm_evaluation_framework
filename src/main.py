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
            f.write(
                'openai_api_key: "YOUR_OPENAI_API_KEY_HERE"\n'
                '\n'
                '# Basic Authentication Configuration\n'
                'auth:\n'
                '  username: "user"\n'
                '  password: "password123"\n'
            )
        print(f"\n--- IMPORTANT ---")
        print(f"Created a placeholder config file at {config_file_path}.")
        print("Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual OpenAI API key.")
        print("You can also change the default 'username' and 'password' in the 'auth' section.")
        print("The application cannot start without a valid OpenAI API key or if auth credentials are not set.")
        print(f"--- IMPORTANT ---\n")
        sys.exit(1) # Exit to prompt user to update config

    # Check if the API key or auth credentials are still the placeholder values
    try:
        with open(config_file_path, 'r') as f:
            config_content = yaml.safe_load(f)
            openai_key = config_content.get("openai_api_key")
            auth_config = config_content.get("auth", {})
            auth_username = auth_config.get("username")
            auth_password = auth_config.get("password")

            if openai_key == "YOUR_OPENAI_API_KEY_HERE":
                print(f"\n--- IMPORTANT ---")
                print(f"Your OpenAI API key in {config_file_path} is still the placeholder value.")
                print("Please update it with your actual key.")
                print(f"--- IMPORTANT ---\n")
                sys.exit(1)
            
            if not auth_username or not auth_password: # Basic check for empty or missing auth fields
                 print(f"\n--- WARNING ---")
                 print(f"Authentication credentials in {config_file_path} are incomplete or missing.")
                 print("Please ensure 'username' and 'password' are set under 'auth' section.")
                 print(f"--- WARNING ---\n")
            elif auth_username == "user" and auth_password == "password123":
                 print(f"\n--- NOTE ---")
                 print(f"Using default authentication credentials in {config_file_path}.")
                 print("Consider changing these for a production environment.")
                 print(f"--- NOTE ---\n")

    except Exception as e:
        print(f"Warning: Could not read or parse config file during startup check: {e}")
        # Continue starting, as the app.py will also check it.

    print("Starting LLM Evaluation Framework API...")
    print("Access FastAPI interactive documentation (Swagger UI) at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
    uvicorn.run(app, host="127.0.0.1", port=8000)

