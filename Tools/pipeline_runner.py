import os
import builtins
import runpy
import tempfile
from pathlib import Path
import nbformat
from nbconvert import PythonExporter
import time

# keep track of time
start = time.time()


# --------------------
# EDIT THESE
# --------------------
DISEASE = input("Disease: ")

notebooks = [
    "Method/hypergraph_overview.ipynb",
    "Gen_Hypergraph/DrugGeneHypergraph.ipynb",
    "Method/Diffusion_Distance_Based_Clustering.ipynb",
    "Method/Clustering_Result_Analysis.ipynb",
    "Method/important_terms_analysis.ipynb",
    "Method/disease_jaccard_similarity.ipynb",
    "Method/community_pie_chart_multiple.ipynb"
]

answers = iter([DISEASE] * len(notebooks))

def fake_input(prompt=""):
    ans = next(answers)
    print(f"{prompt}{ans}")
    return ans

real_input = builtins.input
builtins.input = fake_input

try:
    for nb in notebooks:
        nb = Path(nb).resolve()
        nb_dir = nb.parent
        print(f"\n=== Running {nb} (cwd -> {nb_dir}) ===")

        old_cwd = os.getcwd()
        os.chdir(nb_dir)
        try:
            nb_node = nbformat.read(nb, as_version=4)
            code, _ = PythonExporter().from_notebook_node(nb_node)

            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
                f.write(code.encode("utf-8"))
                pyfile = f.name

            runpy.run_path(pyfile, run_name="__main__")
        finally:
            os.chdir(old_cwd)

finally:
    builtins.input = real_input
    
    
# record time
end = time.time()
print(f"Time to complete: {end - start:.2f} seconds")
