# config.py
# Ownquesta AI Agent — Knowledge Base & Configuration (Updated for Actual Tutorial)
# Place this file in: backend/questa/config.py

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# OPENAI CONFIGURATION
# ─────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("QUESTA_MODEL", "gpt-4o-mini")
MAX_TOKENS: int = int(os.getenv("QUESTA_MAX_TOKENS", "800"))
TEMPERATURE: float = float(os.getenv("QUESTA_TEMPERATURE", "0.7"))
MAX_HISTORY: int = 10  # Number of past messages to keep in memory

# ─────────────────────────────────────────────
# AGENT IDENTITY
# ─────────────────────────────────────────────
AGENT_NAME = "Questa"
AGENT_ROLE = "Official Ownquesta AI Assistant"
WELCOME_MESSAGE = (
  "Hi! I'm **Questa**, your Ownquesta guide 👋 "
  "Ask me about the real app flow, AutoML Playground, uploads, "
  "model training, testing, or downloads."
)

SUGGESTED_QUESTIONS = [
  "How does Ownquesta work from start to finish?",
  "What happens after I click Analyse?",
  "What is the AutoML Playground?",
  "What is Easy Mode vs Code Mode?",
  "How do I upload my dataset?",
  "How do I test my model?",
  "Do I need coding knowledge?",
  "How long does training take?",
]

APP_KNOWLEDGE = [
  {
    "keywords": {"how does ownquesta work", "what is ownquesta", "start to finish", "workflow"},
    "answer": (
      "Ownquesta follows this 10-step app flow:\n"
      "1. Home page.\n"
      "2. Sign in / register.\n"
      "3. Authenticated home (/home).\n"
      "4. Dashboard.\n"
      "5. AutoML Playground mode selection.\n"
      "6. Upload dataset + target column.\n"
      "7. Auto analysis + model suggestions.\n"
      "8. Prediction test + accuracy review.\n"
      "9. Payment / checkout (when export is locked).\n"
      "10. Model or Python script export."
    ),
    "context": "Ownquesta is a no-code AutoML app with a public home page, authentication flow, /home onboarding screen, dashboard, AutoML Playground, model testing, and export/download actions.",
  },
  {
    "keywords": {"authenticated home", "welcome page", "go to dashboard", "after sign in"},
    "answer": (
      "After sign in, you land on the authenticated home page at /home. "
      "It greets you by name and gives you a single Go to Dashboard button, so you can move straight into the workspace."
    ),
    "context": "The /home page is the authenticated home screen. It is not a separate product area; it is the post-login onboarding screen before the dashboard.",
  },
  {
    "keywords": {"automl playground", "what is the automl playground", "lab"},
    "answer": (
      "The AutoML Playground is your ML workspace. Easy Mode is the no-code path: upload a CSV or Excel file, pick the target column if you know it, click Analyse, review the suggested models, and train one. Code Mode opens a notebook-style workflow for users who want custom Python."
    ),
    "context": "The AutoML Playground supports Easy Mode and Code Mode, accepts CSV/XLS/XLSX uploads, and includes the ML Agent panel, EDA, model suggestions, training, prediction testing, and export actions.",
  },
  {
    "keywords": {"easy mode", "code mode", "difference between easy mode and code mode"},
    "answer": (
      "Easy Mode is best if you want point-and-click ML. Code Mode is for Python users who want notebook cells, custom code, and more control. Both modes use the same uploaded dataset and agent guidance."
    ),
    "context": "Easy Mode is the default no-code workflow. Code Mode is a Jupyter-style notebook with agent-generated cells and optional custom Python.",
  },
  {
    "keywords": {"upload my dataset", "upload csv", "upload excel", "how do i upload"},
    "answer": (
      "Open the AutoML Playground, stay in Easy Mode, and use the Upload CSV / Excel button in the right ML Agent panel. Ownquesta accepts CSV, XLSX, and XLS files. After the upload finishes, you can set the target column and click Analyse."
    ),
    "context": "Dataset upload happens in the right ML Agent panel inside AutoML Playground. The target column is optional but helps the model know what to predict.",
  },
  {
    "keywords": {"what happens after i click analyse", "after analyse", "eda", "what is eda"},
    "answer": (
      "After Analyse, Ownquesta checks missing values, applies feature engineering, and runs EDA. In Easy Mode you see three auto-generated charts, Adjust Settings, and Top 3 Recommended Models. In Code Mode you see generated notebook cells, outputs, and the same model suggestions."
    ),
    "context": "Analysis includes missing-value checks, feature engineering notes, EDA charts, and ranked model recommendations.",
  },
  {
    "keywords": {"how do i test my model", "test my model", "run prediction"},
    "answer": (
      "After training, use the Test Your Model form. Fill in the feature values, click Run Prediction, and the app returns the predicted result plus a confidence score or evaluation output."
    ),
    "context": "Testing uses the form shown after training completes. It is part of the final results screen in both Easy Mode and Code Mode.",
  },
  {
    "keywords": {"do i need coding knowledge", "coding knowledge", "no code"},
    "answer": (
      "No. Easy Mode is built for non-coders and handles the full workflow for you. Code Mode is optional if you want to inspect or edit the generated Python."
    ),
    "context": "Easy Mode is the no-code path; Code Mode is optional for developers.",
  },
  {
    "keywords": {"how long does training take", "training take", "how much time"},
    "answer": (
      "Training time depends on dataset size. Small files usually take about 1 to 5 minutes, medium datasets about 5 to 15 minutes, and larger datasets about 15 to 45 minutes."
    ),
    "context": "Training time depends on data size. Bigger datasets and more complex models take longer.",
  },
  {
    "keywords": {"what if accuracy is low", "low accuracy", "accuracy is low"},
    "answer": (
      "If accuracy is low, try a different model, adjust the test split or cross-validation settings, or upload more data. Very small datasets usually need more rows before the model becomes reliable."
    ),
    "context": "When accuracy is weak, the dashboard supports retraining with different settings and model choices.",
  },
  {
    "keywords": {"download my model", "python script", "download model", "export"},
    "answer": (
      "When training finishes, use the Download Model button to save the trained model and the Python Script button to export the full pipeline as runnable code."
    ),
    "context": "The results screen includes Download Model and Python Script actions for export.",
  },
]


def normalize_question(value: str) -> str:
  return " ".join("".join(char.lower() if char.isalnum() else " " for char in value).split())


def get_local_answer(message: str) -> str | None:
  normalized = normalize_question(message)
  for entry in APP_KNOWLEDGE:
    if any(keyword in normalized for keyword in entry["keywords"]):
      return entry["answer"]
  return None


def get_relevant_context(message: str) -> str:
  normalized = normalize_question(message)
  matches = []

  for entry in APP_KNOWLEDGE:
    if any(keyword in normalized for keyword in entry["keywords"]):
      matches.append(entry["context"])

  if matches:
    return "\n".join(matches[:3])

  return (
    "Ownquesta is a no-code AutoML app with this flow: Home -> Sign In/Register -> Authenticated Home (/home) -> Dashboard -> AutoML Playground -> Upload/Target -> Analyse -> Train/Test -> Payment -> Export. "
    "Use these concrete app pages and features when answering."
  )

# ─────────────────────────────────────────────
# SYSTEM PROMPT — ACTUAL OWNQUESTA TUTORIAL WORKFLOW
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Questa, the official Ownquesta assistant.
You must answer using the exact web app flow below, with product-accurate wording.

CANONICAL PRODUCT FLOW (10 STEPS):
1. Home page.
2. Sign in / register.
3. Authenticated home (/home).
4. Dashboard.
5. AutoML Playground mode selection (Easy Mode or Code Mode).
6. Upload dataset + choose target column.
7. Auto analysis + model suggestions.
8. Prediction test + accuracy review.
9. Payment / checkout (only when export/download is locked).
10. Model export / Python script export.

IMPORTANT TERMINOLOGY RULES:
- Use "AutoML Playground" (not "Lab Playground").
- Treat step 9 (payment) and step 10 (export) as separate steps.
- If the user asks for start-to-finish guidance, return all 10 steps.

PRODUCT FACTS TO USE:
- CSV/XLS/XLSX uploads are supported.
- Easy Mode is no-code and guided.
- Code Mode supports notebook-style custom Python.
- Analysis includes EDA insights and top model recommendations.
- Users can test predictions after training.
- Export buttons provide model download and Python script export.

STYLE:
- Friendly, clear, and practical.
- Use short numbered lists for procedural answers.
- Avoid jargon unless the user asks for technical depth.
- Never invent features that are not in Ownquesta.

OUT-OF-SCOPE HANDLING:
If a question is unrelated to Ownquesta, briefly redirect:
"I can help best with Ownquesta workflows and features. Ask me anything about setup, AutoML Playground, analysis, training, payment, or export." 
"""