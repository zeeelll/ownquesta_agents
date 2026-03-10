from typing import Optional
from typing_extensions import TypedDict


class MLState(TypedDict):
    session_id: str
    stage: str                       # current stage name
    dataset_loaded: bool
    target_column: Optional[str]
    problem_type: Optional[str]
    last_output: str                 # stdout from last execution
    last_code: str                   # code generated for last stage
    finished: bool
    uploaded_file_path: Optional[str]   # posix path to uploaded dataset
    uploaded_filename: Optional[str]    # original filename (for extension detection)
