import os
import builtins
import runpy
import tempfile
from pathlib import Path
import json
import nbformat
from nbconvert import PythonExporter
import time

# keep track of time
start = time.time()


# --------------------
# EDIT THESE
# --------------------
DISEASE = input("Disease: ")
SNOMED_id = input("SNOMED id: ")

# save customized id
path = "Data/disease_to_snomed_id.json"

try:
    with open(path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"{path} does not exist")

data[DISEASE] = int(SNOMED_id)

with open(path, "w") as f:
    json.dump(data, f, indent=2)



notebooks = [
    # hypergraph construction
    "Gen_Hypergraph/DrugGeneHypergraph.ipynb",
    # create communities
    "Methods/DDBC/Diffusion_Distance_Based_Clustering.ipynb",
    # generate important terms
    "Methods/DDBC/Clustering_Result_Analysis.ipynb",
    # analyze important terms, generate table
    "Methods/DDBC/important_terms_analysis.ipynb",
    # make community similarity graph
    "Methods/DDBC/community_similarity_graph.ipynb",
    # hypergraph stat
    "Tools/hypergraph_overview.ipynb",
]

answers = iter([
    DISEASE,
    DISEASE,
    DISEASE,
    DISEASE,
    DISEASE,
    DISEASE,
])

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
