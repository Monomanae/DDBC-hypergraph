import numpy as np
from tqdm import tqdm
def random_talk_with_restart(num_iterations,
                             restart_prob, 
                             binary_matrix, 
                             weighted_matrix, 
                             inverse_diag_node_degree_matrix, 
                             inverse_diag_edge_degree_matrix,
                             v0 = None,
                             bias_constant_vector = None):
    num_genes = binary_matrix.shape[0]
    if (v0 is None):  v0 = np.ones(num_genes) / (num_genes) 
    if (bias_constant_vector is None): bias_constant_vector = np.zeros(num_genes)

    v_curr = v0.copy()  # Start with uniform probability
    transition_matrix = inverse_diag_node_degree_matrix @ binary_matrix @ inverse_diag_edge_degree_matrix @ weighted_matrix.T
    transition_matrix = transition_matrix.T

    distance_list = []

    for k in tqdm(range(num_iterations), desc="Random Walk Progress"):
        # Store previous probability vector
        v_prev = v_curr.copy()

        # Matrix multiplication for transition
        v_curr = (1-restart_prob) * (transition_matrix @ v_prev) + restart_prob * v0 + bias_constant_vector

        # Normalize v_curr to avoid overflow
        v_curr /= v_curr.sum()

        # Calculate distance
        distance = np.sum(np.abs(v_prev - v_curr))
        distance_list.append(distance)

        unsorted = v_curr
        # Sort importance scores in descending order
        importance_scores = np.argsort(v_curr)[::-1]
        importance_values = v_curr[importance_scores]

    # Return importance scores and distance values
    return {"Importance": list(zip(importance_scores, importance_values)), "Distance": distance_list, "unsorted": unsorted}

def random_talk_with_restart(num_iterations,
                             restart_prob, 
                             transition_matrix,
                             v0 = None,
                             bias_constant_vector = None,
                             tqdm_enable = True):
    num_genes = transition_matrix.shape[0]
    if (v0 is None):  v0 = np.ones(num_genes) / (num_genes) 
    if (bias_constant_vector is None): bias_constant_vector = np.zeros(num_genes)

    v_curr = v0.copy()  # Start with uniform probability
    distance_list = []

    if (tqdm_enable):
        iterator = tqdm(range(num_iterations), desc="Random Walk Progress")
    else:
        iterator = range(num_iterations)
    for k in iterator:
        # Store previous probability vector
        v_prev = v_curr.copy()

        # Matrix multiplication for transition
        v_curr = (1-restart_prob) * (transition_matrix @ v_prev) + restart_prob * v0 + bias_constant_vector

        # Normalize v_curr to avoid overflow
        v_curr /= v_curr.sum()

        # Calculate distance
        distance = np.sum(np.abs(v_prev - v_curr))
        distance_list.append(distance)

        unsorted = v_curr
        # Sort importance scores in descending order
        importance_scores = np.argsort(v_curr)[::-1]
        importance_values = v_curr[importance_scores]

    # Return importance scores and distance values
    return {"Importance": list(zip(importance_scores, importance_values)), "Distance": distance_list, "unsorted": unsorted}

def build_transition_matrix(binary_matrix, 
                            weighted_matrix, 
                            inverse_diag_node_degree_matrix, 
                            inverse_diag_edge_degree_matrix):
    transition_matrix = inverse_diag_node_degree_matrix @ binary_matrix @ inverse_diag_edge_degree_matrix @ weighted_matrix.T
    # Check if row sums to 1
    return transition_matrix.T.tocsr()
