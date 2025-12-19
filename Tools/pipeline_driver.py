import subprocess
import sys
from pathlib import Path
from datetime import datetime

python = sys.executable
script = Path("Methods/DDBC/pipeline_runner.py")

disease_id_dict = {
  "BIPOLAR": 13746004,
  "SCHIZOPHRENIA": 58214004,
}

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

env = dict(**__import__("os").environ)
env["MPLBACKEND"] = "Agg"   # 👈 disables plot windows

for key, val in disease_id_dict.items():
    inp = f"{key}\n{val}\n"
    log_file = log_dir / f"{key}.log"

    with open(log_file, "w") as f:
        f.write(f"=== Run started {datetime.now()} ===\n")
        f.write(f"Disease: {key}\nSNOMED ID: {val}\n")
        f.write("=" * 40 + "\n\n")

        subprocess.run(
            [python, "-u", script],
            input=inp,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,            # 👈 important
        )
