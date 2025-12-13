import subprocess
from langchain_core.tools import tool
import os

def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("\n", 1)[0]
    return code.strip()

@tool
def run_code(code: str) -> dict:
    """
    Executes Python code in a temporary file using uv.
    """
    try:
        code = strip_code_fences(code)

        filename = "runner.py"
        os.makedirs("LLMFiles", exist_ok=True)

        with open(os.path.join("LLMFiles", filename), "w") as f:
            f.write(code)

        proc = subprocess.Popen(
            ["uv", "run", filename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="LLMFiles"
        )

        stdout, stderr = proc.communicate()

        if len(stdout) > 10_000:
            stdout = stdout[:10_000] + "...truncated"
        if len(stderr) > 10_000:
            stderr = stderr[:10_000] + "...truncated"

        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": proc.returncode
        }

    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }
