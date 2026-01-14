import subprocess
import sys
from pathlib import Path
from datetime import datetime

python = sys.executable
script = Path("Methods/DDBC/pipeline_runner.py")
log_title = input("Log title: ")

disease_id = [
  "BIPOLAR",
  "SCHIZOPHRENIA",
  "PARKINSON",
  "SCHIZOPHRENIA",
  "DIABETES",
  "LEUKEMIA",
  "BREASTCANCER",
  "NONE"
]

log_dir = Path(f"logs/{log_title}")
log_dir.mkdir(exist_ok=True)

env = dict(**__import__("os").environ)
env["MPLBACKEND"] = "Agg"   # 👈 disables plot windows

for key in disease_id:
    log_file = log_dir / f"{key}.log"

    with open(log_file, "w") as f:
        f.write(f"=== Run started {datetime.now()} ===\n")
        f.write(f"Disease: {key}")
        f.write("=" * 40 + "\n\n")

        subprocess.run(
            [python, "-u", script],
            input=key+ "\n",
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,            # 👈 important
        )
