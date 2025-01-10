import sys
import subprocess

if __name__ == "__main__":
    filtered_args = [arg for arg in sys.argv[1:] if not arg.startswith("--local_rank")]
    command = [sys.executable, "/workspace/optimum-habana/examples/image-to-text/run_pipeline.py"] + filtered_args
    result = subprocess.run(command)
    sys.exit(result.returncode)