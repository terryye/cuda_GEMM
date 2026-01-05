import modal
from modal import FilePatternMatcher
import os


script_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(script_dir, ".."))

image = (
    modal.Image.from_registry(f"nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
        .env({
            "NCCL_DEBUG": "INFO",
        })
        .add_local_dir(".", remote_path="/root")
        .add_local_dir(root_dir + "/src/util", remote_path="/root/util")
        .add_local_dir(script_dir, remote_path="/root/script")
)

app = modal.App("nvcc")

@app.function(image=image, gpu="A100-40gb", timeout=300)
def compile_and_run_cuda(code_path: str):
    import subprocess

    subprocess.run(["nvcc", "-DCUDA=1", "-g", "-G", "-rdc=true", "-arch=native", "-I/root",
                    code_path, "-o", "output.bin"],
                   text=True,  check=True)
    subprocess.run([ "./output.bin"], text=True, check=True)