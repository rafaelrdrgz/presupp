"""Host the demo on Modal (see apps/demo/README.md).

    modal deploy apps/demo/modal_app.py

The model folder is uploaded with the code. One container stays up, capped at half a core and
1.5 GiB.
"""

from pathlib import Path

import modal

DEMO = Path(__file__).resolve().parent
ROOT = DEMO.parent.parent
MODEL = ROOT / "models" / "presupp-roberta-v3"
CACHE = ["**/__pycache__", "**/*.pyc"]
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(DEMO / "requirements.txt"))
    .pip_install("gradio==5.50.0")
    .env({"PRESUPP_MODEL_DIR": "/model", "PYTHONPATH": "/root/demo:/root",
          "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "TRANSFORMERS_VERBOSITY": "error"})
    .add_local_dir(ROOT / "src" / "presupp", "/root/presupp", ignore=CACHE)
    .add_local_dir(DEMO, "/root/demo", ignore=CACHE + ["modal_app.py"])
    .add_local_dir(MODEL, "/model")
)

app = modal.App("presupposition-finder")


# web_server, not asgi_app: asgi_app added ~200 ms per request. One container because gradio
# keeps its queue in memory.
@app.function(image=image, cpu=(0.125, 0.5), memory=(1024, 1536), min_containers=1,
              max_containers=1, timeout=600)
@modal.concurrent(max_inputs=100)
@modal.web_server(PORT, startup_timeout=300, label="presupposition-finder")
def web():
    import subprocess
    import sys

    subprocess.Popen([sys.executable, "-m", "uvicorn", "--factory", "app:asgi_app",
                      "--host", "0.0.0.0", "--port", str(PORT)], cwd="/root/demo")
