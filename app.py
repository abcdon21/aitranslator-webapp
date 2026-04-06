from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

API_URL = "https://router.huggingface.co/hf-inference/models/Helsinki-NLP/opus-mt-en-hi"


def load_local_env():
    """Load key=value pairs from a local .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Do not override variables that already exist in the environment.
            os.environ.setdefault(key, value)


load_local_env()


def get_hf_token():
    """Return the first configured Hugging Face token from common env var names."""
    token = os.environ.get("HF_API_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        return None

    token = token.strip()
    # Support users who accidentally paste the full header value.
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def translation(data):
    token = get_hf_token()
    if not token:
        return "Error: Hugging Face token not found. Set HF_API_TOKEN/HF_TOKEN in environment or in a .env file."

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(API_URL, headers=headers, json={"inputs": data})

    if response.status_code == 401:
        return "Error: Invalid Hugging Face token. Please check HF_API_TOKEN/HF_TOKEN."

    if not response.ok:
        return f"Error: API request failed ({response.status_code}) - {response.text}"

    result = response.json()

    if isinstance(result, list):
        return result[0]["translation_text"]
    else:
        return "Error: " + str(result)

@app.route('/', methods=['GET', 'POST'])
def index():
    translated_text = ""
    if request.method == 'POST':
        data = request.form['data']
        translated_text = translation(data)

    return render_template('index.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)