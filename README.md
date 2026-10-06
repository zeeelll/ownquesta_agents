# OwnQuesta Agents

The AI agent service behind [Ownquesta](https://github.com/zeeelll/ownquesta) — an AutoML platform where junior ML engineers build models from scratch with AI agents guiding them.

## 🧠 What this repo does

Using **LangChain** and other **Generative AI** technologies, this service adds an AI layer on top of the AutoML workflow. The agents can:

- **Understand the ML problem** — read the dataset and task, and explain what kind of model and approach fits
- **Generate code** — produce the ML code for each step (preprocessing, training, evaluation) instead of the user copying it from another AI tool
- **Explain the code** — break down what the generated code does and why, so users learn how the model is built rather than just getting a result

The main agent today is the **ML Assistant**, exposed as a FastAPI sub-app at `/ml-assistant`. New agent features for Ownquesta are added in this repository.

---

## ⚙️ Run instructions

Quick steps to run the agent API locally and use the ML Assistant mounted at `/ml-assistant`.

### 1) Create and activate a Python environment (recommended)

Windows (PowerShell):
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (cmd):
```
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies
```
pip install -e .
```

### 3) Set your OpenAI API key (optional, required for `/ml-assistant/chat`)

PowerShell:
```
$env:OPENAI_API_KEY = "sk-..."
```

cmd:
```
set OPENAI_API_KEY=sk-...
```

Or create a `.env` file and load it before starting the server.

### 4) Run uvicorn from the `ownquesta_agents` folder
```
uv run uvicorn main:app --reload
```

### 5) Access endpoints

| Endpoint | Method | URL |
|----------|--------|-----|
| Service metadata | GET | http://127.0.0.1:8000/meta.json |
| ML Assistant base | GET | http://127.0.0.1:8000/ml-assistant |
| ML Assistant health | GET | http://127.0.0.1:8000/ml-assistant/health |
| Chat | POST | http://127.0.0.1:8000/ml-assistant/chat |

Example chat request body:
```json
{"message": "hello"}
```

## 🛠️ Troubleshooting

If you don't see `/ml-assistant`, restart the server after confirming the files exist in `ml_assistant_agent/` and that `main.py` mounts the sub-app.
