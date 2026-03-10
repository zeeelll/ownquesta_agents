from pathlib import Path
from state import MLState


def _load_call(file_path: str, filename: str) -> str:
    """Return the correct pandas read call based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return f'pd.read_excel("{file_path}")'
    return f'pd.read_csv("{file_path}")'


def codegen(state: MLState) -> dict:
    """
    Generates Python code for the current stage.
    Updates state field: last_code.
    """
    stage    = state["stage"]
    target   = state.get("target_column") or "label"
    fp       = state.get("uploaded_file_path") or "dataset.csv"
    fname    = state.get("uploaded_filename")  or "dataset.csv"

    if stage == "load_dataset":
        code = f"""\
import pandas as pd

df = {_load_call(fp, fname)}
print(df.shape)
print(df.head())
print(df.dtypes)
"""

    elif stage == "inspect_dataset":
        code = """\
print(df.describe(include="all"))
"""

    elif stage == "train_baseline":
        code = f"""\
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

target = "{target}"
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print("trained")
"""

    elif stage == "evaluate":
        code = """\
from sklearn.metrics import accuracy_score

pred = model.predict(X_test)
print("accuracy", accuracy_score(y_test, pred))
"""

    else:
        code = f"# Unknown stage: {stage}\n"

    return {"last_code": code}
