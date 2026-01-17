import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MAX_FILE_SIZE_MB: int = 100
    MIN_ROWS_FOR_ML: int = 50
    MAX_MISSING_PERCENT: float = 50.0
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY.startswith("sk-"))

settings = Settings()