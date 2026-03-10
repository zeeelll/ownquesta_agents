import sys
import json
import io
import ast
import traceback
import base64

_globals = {}


def _auto_print(code: str, buf: io.StringIO) -> None:
    """Jupyter-like behaviour: if the last statement is an expression and
    nothing was printed, evaluate it and write its repr to *buf*.
    Handles pandas DataFrames / Series with pretty-print.
    """
    try:
        tree = ast.parse(code)
        if not tree.body:
            return
        last = tree.body[-1]
        if not isinstance(last, ast.Expr):
            return
        # Compile the last expression in 'eval' mode
        expr_code = compile(ast.Expression(body=last.value), "<cell>", "eval")
        result = eval(expr_code, _globals)
        if result is None:
            return
        # Pretty-print pandas objects
        try:
            import pandas as _pd
            if isinstance(result, (_pd.DataFrame, _pd.Series)):
                buf.write(result.to_string() + "\n")
                return
        except ImportError:
            pass
        # Pretty-print numpy arrays
        try:
            import numpy as _np
            if isinstance(result, _np.ndarray):
                buf.write(_np.array2string(result, threshold=200) + "\n")
                return
        except ImportError:
            pass
        buf.write(repr(result) + "\n")
    except Exception:
        pass  # auto-print is best-effort; never crash the cell


def _capture_charts() -> list[str]:
    """Capture all open matplotlib/seaborn figures as base64-encoded PNG strings.
    Returns an empty list if matplotlib is not imported or no figures are open.
    """
    charts: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive, file-only backend
        import matplotlib.pyplot as plt
        fig_nums = plt.get_fignums()
        if not fig_nums:
            return charts
        for num in fig_nums:
            fig = plt.figure(num)
            buf = io.BytesIO()
            fig.savefig(
                buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none", dpi=100,
            )
            buf.seek(0)
            charts.append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close("all")
    except Exception:
        pass   # matplotlib may not be installed or no figures exist
    return charts


def run():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            code = payload.get("code", "")
        except Exception:
            sys.stdout.write(json.dumps({"stdout": "", "error": "invalid json", "charts": []}) + "\n")
            sys.stdout.flush()
            continue

        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        error = None
        try:
            exec(compile(code, "<cell>", "exec"), _globals)
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout

        # Jupyter-like: auto-print last expression if nothing was printed
        if not buf.getvalue() and not error:
            _auto_print(code, buf)

        # Capture any matplotlib figures produced by the code
        charts = _capture_charts()

        sys.stdout.write(
            json.dumps({"stdout": buf.getvalue(), "error": error, "charts": charts}) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    run()
