OwnQuesta Agents - Run instructions
=================================

Quick steps to run the agent API locally and use the ML Assistant mounted at `/ml-assistant`.

1) Create and activate a Python environment (recommended)

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

2) Install dependencies
```
pip install -e .
```

3) Set your OpenAI API key (optional, required for `/ml-assistant/chat`)

PowerShell:
```
$env:OPENAI_API_KEY = "sk-..."
```

cmd:
```
set OPENAI_API_KEY=sk-...
```

Or create a `.env` file and load it before starting the server.

4) Run uvicorn from the `ownquesta_agents` folder
```
uv run uvicorn main:app --reload
```

5) Access endpoints

- Service metadata: http://127.0.0.1:8000/meta.json
- ML Assistant base: http://127.0.0.1:8000/ml-assistant
- ML Assistant health: http://127.0.0.1:8000/ml-assistant/health
- Chat (POST JSON): http://127.0.0.1:8000/ml-assistant/chat  body: {"message":"hello"}

If you don't see `/ml-assistant`, restart the server after confirming the files exist in `ml_assistant_agent/` and that `main.py` mounts the sub-app.
