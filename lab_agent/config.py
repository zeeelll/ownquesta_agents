import os

# URL of the running lab-agent service.
# Override via environment variable LAB_AGENT_URL.
LAB_AGENT_URL: str = os.getenv("LAB_AGENT_URL", "http://localhost:8020")

# URL of the running lab-backend (code execution) service.
# Override via environment variable LAB_BACKEND_URL.
LAB_BACKEND_URL: str = os.getenv("LAB_BACKEND_URL", "http://localhost:8010")
