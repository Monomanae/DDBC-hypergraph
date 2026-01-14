import os
import builtins
import runpy
import nbformat
from pathlib import Path
from nbconvert import PythonExporter


def run_notebook_many_times(ipynb_path, input_sets):
    ipynb_path = Path(ipynb_path).resolve()
    nb_dir = ipynb_path.parent

    # convert notebook -> python script (saved next to notebook)
    nb = nbformat.read(str(ipynb_path), as_version=4)
    code, _ = PythonExporter().from_notebook_node(nb)

    py_path = ipynb_path.with_suffix(".py")
    py_path.write_text(code, encoding="utf-8")

    # run multiple times
    for i, inputs in enumerate(input_sets, 1):
        print(f"\n==== RUN {i} ====")

        it = iter(inputs)
        old_input = builtins.input
        old_cwd = os.getcwd()

        builtins.input = lambda prompt="": next(it)

        try:
            os.chdir(nb_dir)              # ✅ important line
            runpy.run_path(str(py_path))
        finally:
            os.chdir(old_cwd)             # restore cwd
            builtins.input = old_input    # restore input()


if __name__ == "__main__":
    run_notebook_many_times(
        "Methods/DDBC/Diffusion_Distance_Based_Clustering.ipynb",
        [
            ["BIPOLAR"],
            ["LEUKEMIA"],
            ["BREASTCANCER"],
            ["DIABETES"],
            ["PARKINSON"],
            ["SCHIZOPHRENIA"]
        ]
    )
