"""Run run_pipeline_null.ipynb repeatedly to build a distribution of null realizations.

Each replicate executes in a fresh kernel, so the several GB the pipeline holds is
released before the next one starts. The notebook stamps its own run_id from the clock
and saves P_t_avg_dgidb_shared_rows_{run_id}.npy, so replicates never overwrite each
other and nothing needs to be passed in from here. The notebook file is read but never
written back to, so its saved outputs stay as they are.

Run it with the Python that runs the notebooks (3.10) -- the kernel is launched with
sys.executable, so a different interpreter gets a kernel without the pipeline's imports:

    C:\\Users\\celem\\AppData\\Local\\Programs\\Python\\Python310\\python.exe run_null_replicates.py [num_replicates]
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

if importlib.util.find_spec("ipykernel") is None:
    sys.exit(f"{sys.executable} has no ipykernel -- see the usage note above.")

NOTEBOOK = Path(__file__).parent / "run_pipeline_null.ipynb"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "num_replicates",
    nargs="?",
    type=int,
    default=20,
    help="how many null realizations to generate (default: 20)",
)
num_replicates = parser.parse_args().num_replicates

for replicate in range(1, num_replicates + 1):
    print(f"=== replicate {replicate}/{num_replicates} ===", flush=True)
    start = time.monotonic()

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=-1,
        kernel_name="python3",
        # Relative paths in the notebook ("output/NULL") resolve against this.
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )

    # One bad draw should not cost the remaining replicates.
    try:
        client.execute()
        print(f"done in {time.monotonic() - start:.0f}s", flush=True)
    except Exception as error:
        print(f"replicate {replicate} failed: {error}", file=sys.stderr, flush=True)
