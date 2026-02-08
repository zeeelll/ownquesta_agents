import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # No limits
    MAX_FILE_SIZE_MB = None
    MIN_ROWS_FOR_ML = None
    MAX_MISSING_PERCENT = None
    
    @property
    def has_openai_key(self) -> bool:
        return bool(self.OPENAI_API_KEY)

settings = Settings()
