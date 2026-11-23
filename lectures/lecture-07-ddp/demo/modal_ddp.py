# modal_ddp.py: run the Lecture 7 demo on 2 x H100 on Modal.
#
#   Smoke test (small sizes, about 2 minutes):  modal run modal_ddp.py
#   Full run (executes the whole notebook):     modal run modal_ddp.py --full
#
# Results are written next to this file, in modal_results_smoke/ or modal_results_full/.
import io
import pathlib
import zipfile

import modal

HERE = pathlib.Path(__file__).parent
app = modal.App("hpml-l7-ddp-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "numpy", "matplotlib", "nbconvert", "nbformat", "ipykernel")
    .add_local_dir(HERE, remote_path="/root/demo",
                   ignore=["modal_results*", "results*", "__pycache__", "*.zip", "*.nsys-rep"])
)


@app.function(image=image, gpu="H100!:2", timeout=40 * 60, memory=32768)
def run(full: bool) -> bytes:
    import os
    import shutil
    import subprocess

    names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, check=True).stdout.split("\n")
    names = [n.strip() for n in names if n.strip()]
    print("GPUs:", names, flush=True)
    if len(names) != 2 or not all("H100" in n for n in names):
        raise RuntimeError(f"expected 2 x H100, got {names}")

    shutil.copytree("/root/demo", "/tmp/work")
    os.chdir("/tmp/work")
    if full:
        cmd = ("jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 "
               "--output HPML_2026_L7_Demo_executed.ipynb HPML_2026_L7_Demo.ipynb")
    else:
        cmd = ("torchrun --standalone --nproc_per_node=2 ddp_demo.py --part all --width 1024 --layers 4 "
               "--batch 64 --steps 3 --warmup 1 --buckets 1,25 --out results_smoke")
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in pathlib.Path(".").rglob("*"):
            if f.is_file() and (f.parts[0].startswith("results") or f.name.endswith("_executed.ipynb")):
                z.write(f)
    return buf.getvalue()


@app.local_entrypoint()
def main(full: bool = False):
    data = run.remote(full)
    out = HERE / ("modal_results_full" if full else "modal_results_smoke")
    out.mkdir(exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    print("saved to", out)
