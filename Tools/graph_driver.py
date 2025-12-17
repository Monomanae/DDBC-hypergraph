import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

python = sys.executable
script = Path("Methods/DDBC/pipeline_runner.py")

with open("../../Data/disease_to_snomed_id.json", "r") as file:
    disease_to_snomed_id = json.load(file)
LIST_OF_DISEASES = list(disease_to_snomed_id.keys())

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

env = dict(**__import__("os").environ)
env["MPLBACKEND"] = "Agg"   # 👈 disables plot windows

for disease in LIST_OF_DISEASES:
    subprocess.run(
        [python, "-u", script],
        input=disease,
        text=True,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
        env=env,            # 👈 important
    )
