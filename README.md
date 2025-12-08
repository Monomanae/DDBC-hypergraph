# Drug Gene Interaction Hypergraph Analysis
---
### Description
---
This is the repo associated with the Disease-Drug-Gene Hypergraph analysis paper. Take a look at Tools/Visualization to see results

### Install
---
Requires `python3.10` 
`pip install -r requirements.txt` 

### 📂 Folder Descriptions
- **`Data/`** - Download data at https://drive.google.com/file/d/1tJJHrBF0fgERfoB9pqtSf_P84vpcuen7/view?usp=sharing

- **`Gen_Hypergraph/`** – Includes scripts for generating the incidence matrices of hypergraphs used in the methods. Each notebook will generate a single layer. Each is weighted with the PPI network.
hypergraphs for a specific disease, Outputs will be under **`Gen_Hypergraph/outputs`**
- **`Methods/`** – Contains the Random Walk with Restart on a single layer, Drug Guided approach, and GNN methods
- **`Tools/`** – Contains various analysis tools such as Saliency and WMRA



