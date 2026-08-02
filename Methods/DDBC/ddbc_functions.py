"""Reusable functions from Diffusion_Distance_Based_Clustering.ipynb."""

import gc
import json
import os
import pickle
import sys
import random
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import igraph as ig
import leidenalg as la
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from IPython import get_ipython
from matplotlib.backends.backend_pdf import PdfPages
from pypdf import PdfReader, PdfWriter
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm


@dataclass
class DDBCConfig:
    """Configuration values used by the original DDBC notebook."""

    disease: str
    average_t: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    interlayer_transition_prob: float = 0.35
    num_neighbors: int = 400
    resolution: float = 1.3

    size_cap: int = 100
    score_cap: float = 0

    n_boots: int = 1000
    check_every: int = 50
    tol: float = 0.01
    patience: int = 2
    sampling_pct: float = 0.8
    leiden_seed: int = 42

    stationary_tol: float = 1e-9
    stationary_maxit: int = 20000
    stationary_seed: int = 0

    output_root: str = "../output"
    dgidb_output_root: str = "../../Gen_Hypergraph/output"
    graphs_root: str = "../../Graphs"

    @property
    def output_directory(self):
        return f"{self.output_root}/{self.disease}/"

    @property
    def dgidb_directory(self):
        return f"{self.dgidb_output_root}/DGIDB_{self.disease}/"

    @property
    def msigdb_directory(self):
        return f"{self.dgidb_output_root}/MSigDB_Full/"

    @property
    def graph_directory(self):
        return f"{self.graphs_root}/{self.disease}"


def load_matrices(config):
    """Load the matrices used by the DGIDB and MSIGDB layers."""

    DGIDB_DIRECTORY = config.dgidb_directory
    MSIGDB_DIRECTORY = config.msigdb_directory

    # MODIFIED: DISEASE="NONE" is the MSIGDB-only pipeline.
    if config.disease == "NONE":
        return {
            "MSIGDB_binary_matrix": sp.load_npz(
                MSIGDB_DIRECTORY + "hypergraph_incidence_matrix_binary.npz"
            ),
            "MSIGDB_weighted_matrix": sp.load_npz(
                MSIGDB_DIRECTORY + "hypergraph_incidence_matrix_weighted.npz"
            ),
            "MSIGDB_gene_weight_diag_matrix": sp.load_npz(
                MSIGDB_DIRECTORY + "gene_weight_diag_matrix.npz"
            ),
            "MSIGDB_diag_node_degree_matrix": sp.load_npz(
                MSIGDB_DIRECTORY + "diag_node_degree_matrix.npz"
            ),
            "MSIGDB_inverse_diag_edge_degree_matrix": sp.load_npz(
                MSIGDB_DIRECTORY + "inverse_diag_edge_degree_matrix.npz"
            ),
        }

    matrices = {
        "DGIDB_binary_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "hypergraph_incidence_matrix_binary.npz"
        ),
        "DGIDB_weighted_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "hypergraph_incidence_matrix_weighted.npz"
        ),
        "DGIDB_gene_weight_diag_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "diag_gene_weight_matrix.npz"
        ),
        "DGIDB_diag_node_degree_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "diag_node_degree_matrix.npz"
        ),
        "DGIDB_inverse_diag_edge_degree_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "inverse_diag_edge_degree_matrix.npz"
        ),
        "DGIDB_inverse_diag_edge_degree_weightless_matrix": sp.load_npz(
            DGIDB_DIRECTORY + "inverse_diag_edge_degree_weightless_matrix.npz"
        ),
        "MSIGDB_binary_matrix": sp.load_npz(
            MSIGDB_DIRECTORY + "hypergraph_incidence_matrix_binary.npz"
        ),
        "MSIGDB_weighted_matrix": sp.load_npz(
            MSIGDB_DIRECTORY + "hypergraph_incidence_matrix_weighted.npz"
        ),
        "MSIGDB_gene_weight_diag_matrix": sp.load_npz(
            MSIGDB_DIRECTORY + "gene_weight_diag_matrix.npz"
        ),
        "MSIGDB_diag_node_degree_matrix": sp.load_npz(
            MSIGDB_DIRECTORY + "diag_node_degree_matrix.npz"
        ),
        "MSIGDB_inverse_diag_edge_degree_matrix": sp.load_npz(
            MSIGDB_DIRECTORY + "inverse_diag_edge_degree_matrix.npz"
        ),
    }
    return matrices


def csr_equal_tol(A, B, atol=1e-8):
    if (
        A.shape != B.shape
        or not np.array_equal(A.indptr, B.indptr)
        or not np.array_equal(A.indices, B.indices)
    ):
        return False
    return np.allclose(A.data, B.data, atol=atol, rtol=0)


def row_stochastic_check(A):
    row_sums = np.array(A.sum(axis=1)).ravel()
    print(row_sums)
    ok = np.all(np.isclose(row_sums, 1.0))
    print("Every row sums to 1?", ok)
    return ok


def is_symmetric(W, tol=1e-12):
    diff = W - W.T
    if len(np.abs(diff.data)) == 0:
        print("Matrix is exactly symmetric.")
        return True
    check = np.all(np.abs(diff.data) < tol)
    print(max(np.abs(diff.data)))
    return check


def degree_array(A, a=1):
    return np.asarray(A.sum(axis=a)).ravel()


def degree_diagonal_matrix(W, a=1):
    d = degree_array(W, a)
    return sp.diags(d, offsets=0, format="csr")


def symmetrically_normalize(W, a=1):
    D = np.asarray(W.sum(axis=a)).ravel()
    D_inv_sqrt = np.zeros_like(D)
    nze = D != 0
    D_inv_sqrt[nze] = 1 / np.sqrt(D[nze])

    W_sym = W.multiply(D_inv_sqrt)
    W_sym = W_sym.multiply(D_inv_sqrt[:, None])
    return W_sym.tocsr()


def big_objects(n=10, min_mb=1):
    """Show the largest objects currently in the notebook namespace."""

    def get_size(obj):
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                return obj.memory_usage(deep=True).sum()
            if sp.issparse(obj):
                return obj.data.nbytes + obj.indptr.nbytes + obj.indices.nbytes
            return sys.getsizeof(obj)
        except Exception:
            return 0

    ip = get_ipython()
    namespace = globals() if ip is None else ip.user_ns
    items = []
    for name, value in namespace.items():
        if name.startswith("_"):
            continue
        size = get_size(value)
        if size > min_mb * 1024**2:
            items.append((name, type(value).__name__, size))

    items.sort(key=lambda item: item[2], reverse=True)
    print(f"{'Variable':30s} {'Type':25s} {'Size (MB)':>10s}")
    print("-" * 70)
    for name, object_type, size in items[:n]:
        print(f"{name:30s} {object_type:25s} {size / 1024**2:10.2f}")


def relative_normality_error(A):
    if sp.issparse(A):
        AH = A.getH()
        AAH = A @ AH
        AHA = AH @ A
        C = AAH - AHA

        num = np.sqrt(np.sum(np.abs(C.data) ** 2))
        den = max(
            np.sqrt(np.sum(np.abs(AAH.data) ** 2)),
            np.sqrt(np.sum(np.abs(AHA.data) ** 2)),
            1e-30,
        )
        return num / den

    AH = A.conj().T
    AAH = A @ AH
    AHA = AH @ A
    return np.linalg.norm(AAH - AHA, "fro") / max(
        np.linalg.norm(AAH, "fro"),
        np.linalg.norm(AHA, "fro"),
        1e-30,
    )


def build_layer_adjacency_matrices(matrices):
    """Build the two within-layer adjacency matrices."""

    # MODIFIED: A dictionary without DGIDB matrices came from DISEASE="NONE".
    if "DGIDB_weighted_matrix" not in matrices:
        H = matrices["MSIGDB_weighted_matrix"]
        W_v = matrices["MSIGDB_gene_weight_diag_matrix"]
        D_v = matrices["MSIGDB_diag_node_degree_matrix"]
        D_e_inv = matrices["MSIGDB_inverse_diag_edge_degree_matrix"]

        d = (D_v @ W_v).diagonal()
        d_inv = np.zeros_like(d)
        nonzero_mask = d > 0
        d_inv[nonzero_mask] = 1.0 / d[nonzero_mask]
        D_v_inv = sp.diags(d_inv)
        MSIGDB_adjacency_matrix = D_v_inv @ H @ D_e_inv @ H.T

        assert row_stochastic_check(MSIGDB_adjacency_matrix)
        return MSIGDB_adjacency_matrix

    H = matrices["DGIDB_weighted_matrix"]
    W_v = matrices["DGIDB_gene_weight_diag_matrix"]
    D_v = matrices["DGIDB_diag_node_degree_matrix"]
    D_e_inv = matrices["DGIDB_inverse_diag_edge_degree_matrix"]

    d = (D_v @ W_v).diagonal()
    d_inv = np.zeros_like(d)
    nonzero_mask = d > 0
    d_inv[nonzero_mask] = 1.0 / d[nonzero_mask]
    D_v_inv = sp.diags(d_inv)
    DGIDB_adjacency_matrix = D_v_inv @ H @ D_e_inv @ H.T

    H = matrices["MSIGDB_weighted_matrix"]
    W_v = matrices["MSIGDB_gene_weight_diag_matrix"]
    D_v = matrices["MSIGDB_diag_node_degree_matrix"]
    D_e_inv = matrices["MSIGDB_inverse_diag_edge_degree_matrix"]

    d = (D_v @ W_v).diagonal()
    d_inv = np.zeros_like(d)
    nonzero_mask = d > 0
    d_inv[nonzero_mask] = 1.0 / d[nonzero_mask]
    D_v_inv = sp.diags(d_inv)
    MSIGDB_adjacency_matrix = D_v_inv @ H @ D_e_inv @ H.T

    assert row_stochastic_check(DGIDB_adjacency_matrix)
    assert row_stochastic_check(MSIGDB_adjacency_matrix)
    return DGIDB_adjacency_matrix, MSIGDB_adjacency_matrix


def randomize_layer_adjacency_matrix(
    A,
    swaps_per_edge=10,
    seed=None,
    show_progress=True,
):
    """Degree-preserving randomization of a row-stochastic transition matrix.

    Off-diagonal entries are rewired with double-edge swaps that exchange the
    *column* labels of two entries: (i,j) and (k,l) become (i,l) and (k,j). Each
    weight therefore never leaves its original row, so every row sum -- and hence
    row-stochasticity -- is preserved exactly with no renormalization. Out-degree
    (entries per row), in-degree (entries per column) and the diagonal
    self-transition weights are all preserved too, while the pairing between
    source and target genes is destroyed.
    """

    A = sp.csr_matrix(A)
    n = A.shape[0]
    diagonal = sp.diags(A.diagonal())

    off_diagonal = (A - diagonal).tocoo()
    off_diagonal.eliminate_zeros()
    rows = off_diagonal.row.astype(np.int64)
    cols = off_diagonal.col.astype(np.int64)
    weights = off_diagonal.data
    num_edges = rows.size

    rng = np.random.default_rng(seed)
    existing = set((rows * n + cols).tolist())

    target_swaps = int(swaps_per_edge * num_edges)
    max_attempts = 100 * target_swaps
    accepted = 0
    attempts = 0
    progress = tqdm(
        total=target_swaps, desc="edge swaps", disable=not show_progress
    )

    while accepted < target_swaps and attempts < max_attempts:
        batch = min(8192, max_attempts - attempts)
        e1s = rng.integers(0, num_edges, batch)
        e2s = rng.integers(0, num_edges, batch)

        for e1, e2 in zip(e1s, e2s):
            attempts += 1
            i, j = rows[e1], cols[e1]
            k, l = rows[e2], cols[e2]
            # Same row would be a no-op; equal targets forbid the swap outright.
            if i == k or j == l:
                continue
            # Swapped targets must not land on the diagonal.
            if i == l or k == j:
                continue

            new1 = i * n + l
            new2 = k * n + j
            if new1 in existing or new2 in existing:
                continue

            existing.discard(i * n + j)
            existing.discard(k * n + l)
            existing.add(new1)
            existing.add(new2)
            cols[e1], cols[e2] = l, j
            accepted += 1
            progress.update(1)
            if accepted == target_swaps:
                break

    progress.close()
    print(f"{accepted} swaps accepted out of {attempts} attempts")

    rewired = sp.coo_matrix((weights, (rows, cols)), shape=A.shape)
    return (rewired + diagonal).tocsr()


def build_interlayer_coupling_matrices(
    DGIDB_adjacency_matrix,
    MSIGDB_adjacency_matrix,
    config,
):
    """Build C12, C21, A12, A21, B12, and B21 exactly as in the notebook."""

    with open(config.dgidb_directory + "gene_to_index.json", "r") as file:
        dgidb = json.load(file)
    with open(config.msigdb_directory + "gene_to_index.json", "r") as file:
        msigdb = json.load(file)

    DGIDB_index_to_gene = {index: gene for gene, index in dgidb.items()}
    MSIGDB_index_to_gene = {index: gene for gene, index in msigdb.items()}
    num_genes_msigdb = len(msigdb)
    num_genes_dgidb = len(dgidb)

    dgidb_to_msigdb_indices_dict = {}
    C12 = np.zeros((num_genes_dgidb, num_genes_msigdb))
    C21 = np.zeros((num_genes_msigdb, num_genes_dgidb))
    B12_array = np.zeros(num_genes_dgidb)
    B21_array = np.zeros(num_genes_msigdb)

    matched = 0
    for gene_dgidb, idx_dgidb in dgidb.items():
        if gene_dgidb in msigdb:
            idx_msigdb = msigdb[gene_dgidb]
            dgidb_to_msigdb_indices_dict[idx_dgidb] = idx_msigdb
            matched += 1
        else:
            dgidb_to_msigdb_indices_dict[idx_dgidb] = None
            print(f"Gene {gene_dgidb} not found in MSIGDB mapping.")

    for idx_dgidb, idx_msigdb in dgidb_to_msigdb_indices_dict.items():
        if idx_msigdb is not None:
            C12[idx_dgidb, :] = (
                MSIGDB_adjacency_matrix.getrow(idx_msigdb).toarray().ravel()
            )
            C21[idx_msigdb, :] = (
                DGIDB_adjacency_matrix.getrow(idx_dgidb).toarray().ravel()
            )
            B12_array[idx_dgidb] = config.interlayer_transition_prob
            B21_array[idx_msigdb] = config.interlayer_transition_prob

    A12_array = 1 - B12_array
    A21_array = 1 - B21_array
    B12 = sp.diags(B12_array)
    B21 = sp.diags(B21_array)
    A12 = sp.diags(A12_array)
    A21 = sp.diags(A21_array)

    print(matched / len(dgidb), "of DGIDB genes have a match in MSIGDB")

    with open(
        config.output_directory + "dgidb_to_msigdb_indices_dict.json", "w"
    ) as file:
        json.dump(dgidb_to_msigdb_indices_dict, file, indent=4)
    print(
        "Mappings saved to "
        + config.output_directory
        + "dgidb_to_msigdb_indices_dict.json"
    )

    return {
        "dgidb": dgidb,
        "msigdb": msigdb,
        "DGIDB_index_to_gene": DGIDB_index_to_gene,
        "MSIGDB_index_to_gene": MSIGDB_index_to_gene,
        "num_genes_dgidb": num_genes_dgidb,
        "num_genes_msigdb": num_genes_msigdb,
        "dgidb_to_msigdb_indices_dict": dgidb_to_msigdb_indices_dict,
        "C12": C12,
        "C21": C21,
        "A12": A12,
        "A21": A21,
        "B12": B12,
        "B21": B21,
    }
    
def build_interlayer_coupling_matrices_randomized(
    DGIDB_adjacency_matrix,
    MSIGDB_adjacency_matrix,
    config,
    none_prob = 0.015
):
    """Build C12, C21, A12, A21, B12, and B21 exactly as in the notebook."""
    
    filename = "dgidb_to_msigdb_indices_dict"

    with open(config.dgidb_directory + "gene_to_index.json", "r") as file:
        dgidb = json.load(file)
    with open(config.msigdb_directory + "gene_to_index.json", "r") as file:
        msigdb = json.load(file)
        
    num_genes_msigdb = len(msigdb)
    num_genes_dgidb = len(dgidb)
        
    ### RANDOMIZE gene_to_index dictionaries
    random_msigdb_genes_selected = random.sample(list(msigdb.keys()), num_genes_dgidb)
    randomized_dgidb = {
        (f"UNIQUE_GENE_{idx}" if random.random() < none_prob else gene): idx
        for idx, gene in enumerate(random_msigdb_genes_selected)
    }
    
    dgidb = randomized_dgidb   
    ######################################

    DGIDB_index_to_gene = {index: gene for gene, index in dgidb.items()}
    MSIGDB_index_to_gene = {index: gene for gene, index in msigdb.items()}

    dgidb_to_msigdb_indices_dict = {}
    C12 = np.zeros((num_genes_dgidb, num_genes_msigdb))
    C21 = np.zeros((num_genes_msigdb, num_genes_dgidb))
    B12_array = np.zeros(num_genes_dgidb)
    B21_array = np.zeros(num_genes_msigdb)

    matched = 0
    for gene_dgidb, idx_dgidb in dgidb.items():
        if gene_dgidb in msigdb:
            idx_msigdb = msigdb[gene_dgidb]
            dgidb_to_msigdb_indices_dict[idx_dgidb] = idx_msigdb
            matched += 1
        else:
            dgidb_to_msigdb_indices_dict[idx_dgidb] = None
            print(f"Gene {gene_dgidb} not found in MSIGDB mapping.")

    for idx_dgidb, idx_msigdb in dgidb_to_msigdb_indices_dict.items():
        if idx_msigdb is not None:
            C12[idx_dgidb, :] = (
                MSIGDB_adjacency_matrix.getrow(idx_msigdb).toarray().ravel()
            )
            C21[idx_msigdb, :] = (
                DGIDB_adjacency_matrix.getrow(idx_dgidb).toarray().ravel()
            )
            B12_array[idx_dgidb] = config.interlayer_transition_prob
            B21_array[idx_msigdb] = config.interlayer_transition_prob

    A12_array = 1 - B12_array
    A21_array = 1 - B21_array
    B12 = sp.diags(B12_array)
    B21 = sp.diags(B21_array)
    A12 = sp.diags(A12_array)
    A21 = sp.diags(A21_array)

    print(matched / len(dgidb), "of DGIDB genes have a match in MSIGDB")

    with open(
        config.output_directory + f"{filename}.json", "w"
    ) as file:
        json.dump(dgidb_to_msigdb_indices_dict, file, indent=4)
    print(
        "Mappings saved to "
        + config.output_directory
        + f"{filename}.json"
    )

    return {
        "dgidb": dgidb,
        "msigdb": msigdb,
        "DGIDB_index_to_gene": DGIDB_index_to_gene,
        "MSIGDB_index_to_gene": MSIGDB_index_to_gene,
        "num_genes_dgidb": num_genes_dgidb,
        "num_genes_msigdb": num_genes_msigdb,
        "dgidb_to_msigdb_indices_dict": dgidb_to_msigdb_indices_dict,
        "C12": C12,
        "C21": C21,
        "A12": A12,
        "A21": A21,
        "B12": B12,
        "B21": B21,
    }

def build_distinct_gene_mapping(coupling, config):
    filename = "gene_to_index_distinct"
    
    dgidb_to_msigdb_indices_dict = coupling["dgidb_to_msigdb_indices_dict"]
    DGIDB_index_to_gene = coupling["DGIDB_index_to_gene"]
    msigdb = coupling["msigdb"]

    gene_to_index_dgidb_new = {}
    num_additional_rows = 0
    for didx, midx in dgidb_to_msigdb_indices_dict.items():
        if midx is None:
            gene_to_index_dgidb_new[DGIDB_index_to_gene[didx]] = num_additional_rows
            num_additional_rows += 1

    MSIGDB_gene_to_index_new = {
        gene: index + num_additional_rows for gene, index in msigdb.items()
    }
    gene_to_index_distinct = MSIGDB_gene_to_index_new | gene_to_index_dgidb_new

    with open(config.output_directory + f"{filename}.json", "w") as pathway_file:
        json.dump(gene_to_index_distinct, pathway_file, indent=4)
    print(
        "Mappings saved to "
        + config.output_directory
        + f"{filename}.json"
    )
    return gene_to_index_distinct


def build_multilayer_transition_matrix(
    DGIDB_adjacency_matrix,
    MSIGDB_adjacency_matrix,
    coupling,
):
    A = coupling["A12"] @ DGIDB_adjacency_matrix
    B = coupling["B12"] @ coupling["C12"]
    C = coupling["B21"] @ coupling["C21"]
    D = coupling["A21"] @ MSIGDB_adjacency_matrix

    P = sp.bmat([[A, B], [C, D]]).tocsr()
    del A, B, C, D
    for _ in range(3):
        gc.collect()

    assert row_stochastic_check(P)
    support = sp.csr_matrix(P > 0)
    n_components, _ = connected_components(support, directed=False)
    assert n_components == 1
    return P


def stationary_distribution(W, tol=1e-9, maxit=20000, seed=0):
    n = W.shape[0]
    rng = np.random.default_rng(seed)
    pi = rng.random(n) + 1e-12
    pi /= pi.sum()
    for _ in range(maxit):
        pi_next = pi @ W
        if np.linalg.norm(pi_next - pi, 1) < tol:
            break
        pi = pi_next
    return pi / pi.sum()


def build_column_aggregation_matrix(P, coupling, gene_to_index_distinct):
    n1 = coupling["num_genes_dgidb"]
    n2 = coupling["num_genes_msigdb"]
    DGIDB_index_to_gene = coupling["DGIDB_index_to_gene"]
    MSIGDB_index_to_gene = coupling["MSIGDB_index_to_gene"]
    num_distinct_row = len(gene_to_index_distinct)
    total_cols = n1 + n2

    rows = np.empty(total_cols, dtype=int)
    cols = np.empty(total_cols, dtype=int)
    data = np.ones(total_cols, dtype=P.dtype)

    for k in range(n1):
        rows[k] = k
        cols[k] = gene_to_index_distinct[DGIDB_index_to_gene[k]]
    for k in range(n2):
        rows[n1 + k] = n1 + k
        cols[n1 + k] = gene_to_index_distinct[MSIGDB_index_to_gene[k]]

    A_c = sp.csr_matrix(
        (data, (rows, cols)), shape=(P.shape[1], num_distinct_row)
    )
    return A_c


def compute_stationary_layer_weights(pi, coupling):
    n1 = coupling["num_genes_dgidb"]
    n2 = coupling["num_genes_msigdb"]
    mapping = coupling["dgidb_to_msigdb_indices_dict"]

    dgidb_pi_average = pi[0:n1].mean()
    msigdb_pi_average = pi[n1 : n1 + n2].mean()
    print("Average stationary probability for DGIDB genes:", dgidb_pi_average)
    print("Average stationary probability for MSIGDB genes:", msigdb_pi_average)
    print(
        "Ratio of averages (DGIDB/MSIGDB):",
        dgidb_pi_average / msigdb_pi_average,
    )

    wD_list, wM_list = [], []
    for u, v in mapping.items():
        if v is not None:
            pid = pi[u]
            pim = pi[v + n1]
            wd = pid / (pid + pim)
            wm = pim / (pid + pim)
            wD_list.append(wd)
            wM_list.append(wm)
    return wD_list, wM_list


def plot_stationary_layer_weights(wD_list, wM_list, config, run_id = None):
    filename = 'pairwise_weights_comparison_for_duplicated_genes' if run_id is None else f"pairwise_weights_comparison_for_duplicated_genes_{run_id}"
    xaxis = np.arange(len(wD_list))
    plt.figure(figsize=(10, 6))
    plt.plot(xaxis, wD_list, color="blue", label="wD (DGIDB weight)")
    plt.plot(xaxis, wM_list, color="orange", label="wM (MSIGDB weight)")
    plt.xlabel("Gene Index (for genes in both layers)")
    plt.legend()
    plt.title("Layer Weights for Genes in Both Layers")
    plt.savefig(
        f"{config.graph_directory}/{filename}.png"
    )


def build_row_aggregation_matrix(P, pi, coupling, gene_to_index_distinct):
    n1 = coupling["num_genes_dgidb"]
    n2 = coupling["num_genes_msigdb"]
    mapping = coupling["dgidb_to_msigdb_indices_dict"]
    DGIDB_index_to_gene = coupling["DGIDB_index_to_gene"]
    MSIGDB_index_to_gene = coupling["MSIGDB_index_to_gene"]
    num_distinct_row = len(gene_to_index_distinct)

    wD_list2 = []
    total_rows = n1 + n2
    rows = np.empty(total_rows, dtype=int)
    cols = np.empty(total_rows, dtype=int)
    data = np.empty(total_rows, dtype=P.dtype)

    for didx, midx in mapping.items():
        rows[didx] = didx
        cols[didx] = gene_to_index_distinct[DGIDB_index_to_gene[didx]]
        if midx is None:
            data[didx] = 1.0
        else:
            wD = pi[didx] / (pi[didx] + pi[midx + n1])
            wD_list2.append(wD)
            data[didx] = wD

    wM_list2_dict = {}
    for k in range(n2):
        rows[n1 + k] = n1 + k
        cols[n1 + k] = gene_to_index_distinct[MSIGDB_index_to_gene[k]]
        if k in mapping.values():
            corresponding_dgidb_idx = next(
                didx for didx, midx in mapping.items() if midx == k
            )
            wM = pi[n1 + k] / (pi[n1 + k] + pi[corresponding_dgidb_idx])
            wM_list2_dict[corresponding_dgidb_idx] = wM
            data[n1 + k] = wM
        else:
            data[n1 + k] = 1.0

    A_r = sp.csr_matrix(
        (data, (rows, cols)), shape=(P.shape[0], num_distinct_row)
    ).T.tocsr()
    wM_list2 = [wM_list2_dict[k] for k in sorted(wM_list2_dict)]
    return A_r, wD_list2, wM_list2


def apply_exact_power_left(x, t, A_r, P, A_c):
    y = x.copy().astype(np.float64)
    y = y @ A_r
    for _ in range(t):
        y = y @ P
    y = y @ A_c
    return y


def dgidb_aggregate_row_indices(coupling, gene_to_index_distinct, matched_only=False):
    """
    Row indices of every DGIDB gene in the aggregated matrix.

    MODIFIED: matched_only keeps just the DGIDB genes that also live in the MSIGDB
    layer, dropping the DGIDB-only ones. The remaining indices stay in DGIDB index
    order.
    """

    DGIDB_index_to_gene = coupling["DGIDB_index_to_gene"]
    mapping = coupling["dgidb_to_msigdb_indices_dict"]
    n1 = coupling["num_genes_dgidb"]
    return [
        gene_to_index_distinct[DGIDB_index_to_gene[idx]]
        for idx in range(n1)
        if not matched_only or mapping[idx] is not None
    ]


def load_dgidb_aggregate_row_indices(config, matched_only=False):
    with open(
        f"{config.output_directory}/dgidb_to_msigdb_indices_dict.json", "r"
    ) as file:
        dgidb_to_msigdb_indices_dict = json.load(file)

    num_additional_rows = sum(
        1 for midx in dgidb_to_msigdb_indices_dict.values() if midx is None
    )

    idx_list = []
    rank = 0
    for midx in dgidb_to_msigdb_indices_dict.values():
        if midx is None:
            if not matched_only:
                idx_list.append(rank)
            rank += 1
        else:
            idx_list.append(midx + num_additional_rows)
    return idx_list

def compute_and_save_average_transition_matrix(
    P,
    config,
    A_r=None,
    A_c=None,
    dgidb_rows_only=False,
    run_id = None
):
    if config.average_t is None:
        raise ValueError("average_t must be provided.")

    # MODIFIED: No aggregation matrices are used for the MSIGDB-only case.
    msigdb_only = A_r is None and A_c is None

    if (A_r is None) != (A_c is None):
        raise ValueError("A_r and A_c must either both be provided or both be None.")

    # MODIFIED: dgidb_rows_only restricts the computation to the DGIDB genes that also
    # appear in the MSIGDB layer; the DGIDB-only genes are left out. Each row is an
    # independent e_idx A_r P^t A_c product, so skipping rows leaves the remaining ones
    # bit-for-bit identical -- it only avoids the work.
    if dgidb_rows_only and msigdb_only:
        raise ValueError(
            "dgidb_rows_only requires the DGIDB+MSIGDB path; the MSIGDB-only "
            "case has no DGIDB rows."
        )

    if msigdb_only:
        num_rows = P.shape[0]
        num_columns = P.shape[1]
    else:
        num_rows = A_r.shape[0]
        num_columns = A_c.shape[1]

    if dgidb_rows_only:
        target_rows = load_dgidb_aggregate_row_indices(config, matched_only=True)
        if max(target_rows) >= num_rows:
            raise ValueError(
                f"DGIDB row index {max(target_rows)} is out of range for an "
                f"aggregated matrix with {num_rows} rows; the indices and A_r "
                "come from different runs."
            )
        filename = "P_t_avg_dgidb_shared_rows" if run_id is None else f"P_t_avg_dgidb_shared_rows_{run_id}"
    else:
        target_rows = range(num_rows)
        filename = "P_t_avg" if run_id is None else f"P_t_avg_{run_id}"

    P_t_avg = np.zeros((len(target_rows), num_columns), dtype=np.float64)

    # MODIFIED: out_idx is the position in the output, idx the row being computed.
    # They coincide only when every row is computed.
    for out_idx, idx in enumerate(tqdm(target_rows)):
        if msigdb_only:
            idx_row = np.zeros(P.shape[0])
            idx_row[idx] = 1.0
        else:
            e = np.zeros(A_r.shape[0])
            e[idx] = 1.0
            idx_row = e @ A_r

        avg_row = np.zeros(num_columns)
        current_t = 0

        for t in config.average_t:
            while current_t < t:
                idx_row = idx_row @ P
                current_t += 1

            if msigdb_only:
                avg_row += idx_row
            else:
                avg_row += idx_row @ A_c

        avg_row /= len(config.average_t)
        P_t_avg[out_idx] = avg_row

    np.save(f"{config.output_directory}/{filename}.npy", P_t_avg)

    # A row subset is meaningless without knowing which gene each row is, so persist
    # the index list beside the matrix.
    # if dgidb_rows_only:
    #     with open(
    #         f"{config.output_directory}/dgidb_agg_idx_list.json", "w"
    #     ) as file:
    #         json.dump(list(target_rows), file)

    return P_t_avg


def compute_and_save_diffusion_distance(P_t_avg, config, run_id):
    filename = f'ddm_{config.average_t}' if run_id is None else f'ddm_{config.average_t}_{run_id}'
    D_avg = squareform(pdist(P_t_avg, metric="euclidean") ** 2)
    np.save(
        f"{config.output_directory}/{filename}.npy",
        D_avg,
    )
    return D_avg


def load_diffusion_distance(config):
    return np.load(f"{config.output_directory}/ddm_{config.average_t}.npy")


def igraph_from_knn_adjacency(A):
    A.setdiag(0.0)
    A.eliminate_zeros()
    U = sp.triu(A, k=1, format="coo")
    n = A.shape[0]
    graph = ig.Graph(
        n=n,
        edges=list(zip(U.row.tolist(), U.col.tolist())),
        directed=False,
    )
    graph.es["weight"] = U.data.tolist()
    return graph


def leiden_from_igraph(
    graph,
    resolution,
    *,
    n_iterations=-1,
    seed=42,
):
    part = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        n_iterations=n_iterations,
        seed=seed,
        weights="weight",
        resolution_parameter=resolution,
    )
    labels = np.array(part.membership, dtype=np.int32)
    communities = [list(community) for community in part]
    return labels, float(part.quality()), communities


def leiden_from_knn_adjacency(
    A,
    resolution,
    *,
    n_iterations=-1,
    seed=42,
):
    graph = igraph_from_knn_adjacency(A)
    return leiden_from_igraph(
        graph,
        resolution,
        n_iterations=n_iterations,
        seed=seed,
    )


def community_report_onepage(
    labels,
    score,
    out_pdf="leiden_community_report.pdf",
    *,
    extra_text=None,
    bins="auto",
    title="Community Sizes",
    time_steps="N/A",
):
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array of community ids")

    sizes = np.bincount(labels.astype(np.int64, copy=False))
    sizes_sorted = np.sort(sizes)
    n, k = sizes.sum(), sizes.size

    lines = [
        "Leiden Partition Summary",
        "========================",
        f"Nodes (n):           {n:,}",
        f"Time steps:          {time_steps}",
        f"Communities (k):     {k:,}",
        f"Size (min):          {int(sizes_sorted[0]) if k else 0:,}",
        f"Size (median):       {float(np.median(sizes_sorted)) if k else 0.0:.3f}",
        f"Size (mean):         {float(sizes_sorted.mean()) if k else 0.0:.6f}",
        f"Size (max):          {int(sizes_sorted[-1]) if k else 0:,}",
        f"Score:               {score}",
        "",
        "Top 10 largest communities (id: size):",
    ]
    for cid in np.argsort(-sizes)[: min(10, k)]:
        lines.append(f"  {int(cid):5d}: {int(sizes[cid]):,}")
    if extra_text:
        lines += [
            "",
            "Extra:",
            *([extra_text] if isinstance(extra_text, str) else list(extra_text)),
        ]
    summary_text = "\n".join(lines)

    fig = plt.figure(figsize=(8.5, 11), dpi=150)
    ax_text = fig.add_axes([0.06, 0.55, 0.88, 0.40])
    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        summary_text,
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    ax_hist = fig.add_axes([0.10, 0.08, 0.80, 0.38])
    ax_hist.hist(sizes, bins=bins)
    ax_hist.set_xlabel("Community size")
    ax_hist.set_ylabel("Count of communities")
    ax_hist.set_title(f"Distribution of {title}")

    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
        tmp_page = Path(tmpf.name)
    with PdfPages(tmp_page) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    out_pdf = Path(out_pdf)
    tmp_out = out_pdf.with_suffix(out_pdf.suffix + ".tmp")
    writer = PdfWriter()

    if out_pdf.exists():
        with open(out_pdf, "rb") as existing_file:
            reader = PdfReader(existing_file)
            for page in reader.pages:
                writer.add_page(page)

    with open(tmp_page, "rb") as new_file:
        reader_new = PdfReader(new_file)
        for page in reader_new.pages:
            writer.add_page(page)

    with open(tmp_out, "wb") as output_file:
        writer.write(output_file)
    os.replace(tmp_out, out_pdf)

    try:
        os.remove(tmp_page)
    except OSError:
        pass
    return out_pdf


def build_kNN(ddm, k, sym_method="average"):
    tmp = ddm.astype(np.float64, copy=True)
    sigma = np.median(tmp[tmp != 0], overwrite_input=True)
    n = ddm.shape[0]

    rows, cols, vals = [], [], []
    for i in range(n):
        profile = np.exp(-ddm[i] / (2 * sigma))
        idx = np.argpartition(-profile, k + 1)[: k + 1]
        idx = idx[idx != i]
        rows += [i] * k
        cols += list(idx[:k])
        vals += list(profile[idx[:k]])
    adj_mat = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    if sym_method == "average":
        adj_mat = (adj_mat + adj_mat.T).multiply(0.5).tocsr()
    elif sym_method == "union":
        adj_mat = adj_mat.maximum(adj_mat.T)
    elif sym_method == "none":
        pass
    else:
        raise ValueError("sym_method must be 'average' or 'union'")

    kNN_graph = nx.from_scipy_sparse_array(adj_mat)
    return adj_mat, kNN_graph


def run_clustering(kNN_adjacency_matrix, config):
    labels, score, communities = leiden_from_knn_adjacency(
        kNN_adjacency_matrix,
        config.resolution,
        seed=config.leiden_seed,
    )
    pdf_path = f"{config.output_directory}/leiden_report.pdf"
    path = community_report_onepage(
        labels,
        score,
        out_pdf=pdf_path,
        extra_text=[
            f"resolution={config.resolution}",
            f"num_neighbors = {config.num_neighbors}",
            (
                "interlayer_transition_probability = "
                f"{config.interlayer_transition_prob}"
            ),
        ],
        bins=100,
        time_steps=config.average_t,
    )
    print("Wrote:", path)
    print(communities)

    with open(
        f"{config.output_directory}/result_communities.pkl", "wb"
    ) as file:
        pickle.dump(communities, file)
    with open(f"{config.output_directory}/labels.pkl", "wb") as file:
        pickle.dump(labels, file)
    return labels, score, communities


def zscore(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr

    mean = arr.mean()
    std = arr.std(ddof=0)
    if std == 0 or np.isnan(std):
        return np.zeros_like(arr)
    return (arr - mean) / std


def community_central_genes_by_score(
    G,
    community_nodes,
    weight="weight",
    score_cap=1,
):
    C = set(community_nodes)
    H = G.subgraph(C).copy()
    k = {u: H.degree(u, weight=weight) for u in H}
    ks = np.array(list(k.values()), dtype=float)
    zscore_list = zscore(ks)
    Z = dict(zip(H, zscore_list))
    ranked = sorted(H.nodes(), key=lambda u: Z[u], reverse=True)
    return [u for u in ranked if Z[u] >= score_cap]


def select_communities(graph, communities, config):
    communities_selected = []
    for community in communities:
        if len(community) >= config.size_cap:
            important_nodes = community_central_genes_by_score(
                graph,
                community,
                score_cap=config.score_cap,
            )
            a = len(important_nodes)
            b = len(community)
            print(a, b, a / b)
            communities_selected.append(important_nodes)

    with open(
        f"{config.output_directory}/result_communities_selected.pkl", "wb"
    ) as file:
        pickle.dump(communities_selected, file)
    return communities_selected


def run_bootstrap_robustness(kNN_adjacency_matrix, config):
    g_full = igraph_from_knn_adjacency(kNN_adjacency_matrix)
    labels_original, _, _ = leiden_from_igraph(
        g_full,
        config.resolution,
        seed=config.leiden_seed,
    )

    ari_scores = []
    prev_median = None
    streak = 0
    n = kNN_adjacency_matrix.shape[0]

    for b in tqdm(range(config.n_boots)):
        idx = np.random.choice(
            n,
            size=int(config.sampling_pct * n),
            replace=False,
        )
        sim_sub = kNN_adjacency_matrix[np.ix_(idx, idx)]
        g_sub = igraph_from_knn_adjacency(sim_sub)
        labels_sub, _, _ = leiden_from_igraph(
            g_sub,
            config.resolution,
            n_iterations=-1,
            seed=config.leiden_seed,
        )

        ari = adjusted_rand_score(labels_original[idx], labels_sub)
        ari_scores.append(ari)

        if (b + 1) % config.check_every == 0:
            cur_median = float(np.median(ari_scores))
            print(f"Bootstrap {b + 1}: Median ARI = {cur_median:.4f}")

            if (
                prev_median is not None
                and abs(cur_median - prev_median) < config.tol
            ):
                streak += 1
            else:
                streak = 0

            if streak >= config.patience:
                break
            prev_median = cur_median

    with open(
        f"{config.output_directory}/bootstrap_ari_scores.txt", "w"
    ) as file:
        for score in ari_scores:
            file.write(f"{score:.4f}\n")
    return ari_scores


def load_bootstrap_scores(config):
    with open(
        f"{config.output_directory}/bootstrap_ari_scores.txt", "r"
    ) as file:
        return [float(line.strip()) for line in file]