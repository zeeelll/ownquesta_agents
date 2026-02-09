"""Remove __pycache__ directories and .pyc files under the ownquesta_agents folder.

Run this from the repository root or from within the `ownquesta_agents` folder:

python scripts/clean_pycache.py

It is safe: it only removes compiled cache files and directories.
"""
import os
import sys
from pathlib import Path


def remove_pycache(root: Path):
    removed = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # remove .pyc files
        for fname in filenames:
            if fname.endswith('.pyc') or fname.endswith('.pyo'):
                p = Path(dirpath) / fname
                try:
                    p.unlink()
                    removed.append(str(p))
                except Exception:
                    pass

        # remove __pycache__ dirs
        for d in dirnames:
            if d == '__pycache__':
                p = Path(dirpath) / d
                try:
                    for child in p.iterdir():
                        try:
                            child.unlink()
                        except Exception:
                            pass
                    p.rmdir()
                    removed.append(str(p))
                except Exception:
                    pass

    return removed


def main():
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])

    print(f"Cleaning under: {root}")
    removed = remove_pycache(root)
    if removed:
        print("Removed:")
        for r in removed:
            print(" -", r)
    else:
        print("No __pycache__ or .pyc files found.")


if __name__ == '__main__':
    main()
