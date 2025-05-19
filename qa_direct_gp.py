import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import collections
import scipy
from scipy.io import mmread
from dwave.system import DWaveSampler, LeapHybridSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel
from typing import Dict, Tuple

def build_laplacian_matrix(G: nx.Graph) -> np.ndarray:
    A = nx.adjacency_matrix(G).todense()
    D = np.diag([val for _, val in G.degree()])
    return D - A

def get_variable_index(i: int, j: int, k: int) -> int:
    return i * k + j

def build_k_concurrent_qubo(G: nx.Graph, k: int, beta: float, alpha: float, gamma: float) -> BinaryQuadraticModel:
    n = G.number_of_nodes()
    print(f'Number of nodes in the supplied graph:{n}')
    N = n * k
    L = build_laplacian_matrix(G)

    bqm = BinaryQuadraticModel('BINARY')

    # Cut edges penalty
    for i in range(n):
        for j in range(n):
            if i == j: continue
            for p in range(k):
                idx_i = get_variable_index(i, p, k)
                idx_j = get_variable_index(j, p, k)
                weight = beta * L[i, j]
                if weight != 0:
                    bqm.add_interaction(idx_i, idx_j, weight)

    # Balance constraint for partition sizes
    for j in range(k):
        for i1 in range(n):
            idx_i1 = get_variable_index(i1, j, k)
            bqm.add_variable(idx_i1, alpha * (1 - 2 * n / k))
            for i2 in range(i1 + 1, n):
                idx_i2 = get_variable_index(i2, j, k)
                bqm.add_interaction(idx_i1, idx_i2, 2 * alpha)

    #  Uniqueness constraint for each node
    for i in range(n):
        for j1 in range(k):
            idx_j1 = get_variable_index(i, j1, k)
            bqm.add_variable(idx_j1, gamma * (1 - 2))
            for j2 in range(j1 + 1, k):
                idx_j2 = get_variable_index(i, j2, k)
                bqm.add_interaction(idx_j1, idx_j2, 2 * gamma)

    return bqm

def solve_qubo(bqm: BinaryQuadraticModel, num_reads: int = 100, annealing_time: int = 20) -> Dict:
    
    sampler = EmbeddingComposite(DWaveSampler())

    response = sampler.sample(
        bqm,
        num_reads=num_reads,
        annealing_time=annealing_time  # default is typically 20 µs, can be increased up to 2000 µs+
    )

    return response.first.sample

'''def extract_partitions(sample: Dict[int, int], n: int, k: int) -> Dict[int, int]:
    partitions = {}
    for i in range(n):
        for j in range(k):
            idx = get_variable_index(i, j, k)
            if sample.get(idx, 0) == 1:
                partitions[i] = j
                break
    return partitions'''

# Extract partitions and count constraint violations
def extract_partitions_with_violations(sample: Dict[int, int], n: int, k: int, tolerance: float = 0.1) -> Tuple[Dict[int, int], Dict[str, int]]:
    partitions = {}
    over_assigned = 0
    unassigned = 0
    partition_sizes = collections.defaultdict(int)

    for i in range(n):
        assigned = [j for j in range(k) if sample.get(get_variable_index(i, j, k), 0) == 1]
        if len(assigned) != 1:
            if len(assigned) > 1:
                over_assigned += 1
            else:
                unassigned += 1
            partitions[i] = assigned[0] if assigned else -1
        else:
            partitions[i] = assigned[0]
        partition_sizes[partitions[i]] += 1

    expected_size = n // k
    imbalance = sum(1 for size in partition_sizes.values() if abs(size - expected_size) > tolerance*expected_size)

    violation_report = {
        "over_assigned_nodes": over_assigned,
        "unassigned_nodes": unassigned,
        "imbalanced_partitions": imbalance
    }

    return partitions, violation_report

# Count cut edges in the final partitioning
def compute_cut_edges(G: nx.Graph, partitions: Dict[int, int]) -> int:
    return sum(1 for u, v in G.edges() if partitions.get(u) != partitions.get(v))


def draw_partitioned_graph(G: nx.Graph, partitions: Dict[int, int], filename: str = "partitioned_graph.png"):
    pos = nx.spring_layout(G, seed=42)
    color_map = [partitions.get(i, 0) for i in G.nodes()]
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_color=color_map, with_labels=True, cmap=plt.cm.Set2, edge_color='gray')
    plt.title("Partitioned Graph")
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def sweep_partitioning_parameters(
    G: nx.Graph,
    k: int,
    alpha_values: list,
    beta_values: list,
    annealing_times: list,
    read_counts: list,
    gamma: float = 5.0,
    tolerance: float = 0.1,
    fixed_anneal_time: int = 20,
    fixed_num_reads: int = 100
) -> tuple:
    
    from dwave.system import EmbeddingComposite, DWaveSampler

    results_ab = []
    results_ar = []
    n = G.number_of_nodes()

    
    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\n[alpha/beta sweep] alpha={alpha}, beta={beta}")
            bqm = build_k_concurrent_qubo(G, k, beta=beta, alpha=alpha, gamma=gamma)

            sampler = EmbeddingComposite(DWaveSampler())
            sample = sampler.sample(
                bqm,
                annealing_time=fixed_anneal_time,
                num_reads=fixed_num_reads
            ).first.sample

            partitions, violations = extract_partitions_with_violations(sample, n, k, tolerance)
            cut_edges = compute_cut_edges(G, partitions)

            results_ab.append({
                "alpha": alpha,
                "beta": beta,
                "annealing_time": fixed_anneal_time,
                "num_reads": fixed_num_reads,
                "cut_edges": cut_edges,
                **violations
            })

    
    best_result = min(results_ab, key=lambda x: x["cut_edges"])
    best_alpha = best_result["alpha"]
    best_beta = best_result["beta"]
    print(f"\n[Best alpha/beta found] alpha={best_alpha}, beta={best_beta}, cut_edges={best_result['cut_edges']}")

    
    for annealing_time in annealing_times:
        for num_reads in read_counts:
            print(f"[anneal/read sweep] anneal={annealing_time}, reads={num_reads}")
            bqm = build_k_concurrent_qubo(G, k, beta=best_beta, alpha=best_alpha, gamma=gamma)

            sampler = EmbeddingComposite(DWaveSampler())
            sample = sampler.sample(
                bqm,
                annealing_time=annealing_time,
                num_reads=num_reads
            ).first.sample

            partitions, violations = extract_partitions_with_violations(sample, n, k, tolerance)
            cut_edges = compute_cut_edges(G, partitions)

            results_ar.append({
                "alpha": best_alpha,
                "beta": best_beta,
                "annealing_time": annealing_time,
                "num_reads": num_reads,
                "cut_edges": cut_edges,
                **violations
            })

    return results_ab, results_ar


def plot_contours(results_ab: list, results_ar: list):
    
    df_ab = pd.DataFrame(results_ab)
    df_ar = pd.DataFrame(results_ar)

    
    pivot_cut = df_ab.pivot_table(index='alpha', columns='beta', values='cut_edges')
    alphas = pivot_cut.index.values
    betas = pivot_cut.columns.values
    A, B = np.meshgrid(betas, alphas)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    cp1 = axs[0, 0].contourf(B, A, pivot_cut.values, cmap='viridis')
    axs[0, 0].set_title("Cut Edges vs (alpha, beta)")
    axs[0, 0].set_xlabel("beta")
    axs[0, 0].set_ylabel("alpha")
    fig.colorbar(cp1, ax=axs[0, 0])

   
    pivot_imbalance = df_ab.pivot_table(index='alpha', columns='beta', values='imbalanced_partitions')
    cp2 = axs[0, 1].contourf(B, A, pivot_imbalance.values, cmap='plasma')
    axs[0, 1].set_title("Imbalanced Partitions vs (alpha, beta)")
    axs[0, 1].set_xlabel("beta")
    axs[0, 1].set_ylabel("alpha")
    fig.colorbar(cp2, ax=axs[0, 1])

   
    best_row = df_ab.loc[df_ab['cut_edges'].idxmin()]
    best_alpha = best_row['alpha']
    best_beta = best_row['beta']
    df_best = df_ar[(df_ar['alpha'] == best_alpha) & (df_ar['beta'] == best_beta)]

   
    pivot_cut2 = df_best.pivot_table(index='annealing_time', columns='num_reads', values='cut_edges')
    T, R = np.meshgrid(pivot_cut2.columns.values, pivot_cut2.index.values)

    cp3 = axs[1, 0].contourf(R, T, pivot_cut2.values, cmap='viridis')
    axs[1, 0].set_title(f"Cut Edges vs (annealing_time, num_reads)\n@ alpha={best_alpha}, beta={best_beta}")
    axs[1, 0].set_xlabel("num_reads")
    axs[1, 0].set_ylabel("annealing_time")
    fig.colorbar(cp3, ax=axs[1, 0])

   
    pivot_imbalance2 = df_best.pivot_table(index='annealing_time', columns='num_reads', values='imbalanced_partitions')
    cp4 = axs[1, 1].contourf(R, T, pivot_imbalance2.values, cmap='plasma')
    axs[1, 1].set_title(f"Imbalance vs (annealing_time, num_reads)\n@ alpha={best_alpha}, beta={best_beta}")
    axs[1, 1].set_xlabel("num_reads")
    axs[1, 1].set_ylabel("annealing_time")
    fig.colorbar(cp4, ax=axs[1, 1])

    plt.tight_layout()
    plt.savefig("parameter_sweep_contours.png", dpi=300)
    plt.close()


def main():
    import numpy as np
    import networkx as nx

    
    G = nx.erdos_renyi_graph(n=20, p=0.3, seed=42)
    k = 2

    
    alpha_vals = list(np.linspace(2, 10, 5))
    beta_vals = list(np.linspace(0.5, 2, 5))
    anneal_times = list(np.linspace(10, 20, 5))
    read_vals = [int(x) for x in np.linspace(100, 300, 5)]
    print("Read values:", read_vals)

    
    results_ab, results_ar = sweep_partitioning_parameters(
        G,
        k,
        alpha_values=alpha_vals,
        beta_values=beta_vals,
        annealing_times=anneal_times,
        read_counts=read_vals
    )

    
    print("\n--- Sweep over alpha and beta ---")
    for res in results_ab:
        print(res)

    print("\n--- Sweep over annealing time and num_reads ---")
    for res in results_ar:
        print(res)

    with open("results_ab.txt", "w") as f_ab:
        for res in results_ab:
            f_ab.write(json.dumps(res) + "\n")

    with open("results_ar.txt", "w") as f_ar:
        for res in results_ar:
            f_ar.write(json.dumps(res) + "\n")

    
    plot_contours(results_ab,results_ar)

if __name__ == "__main__":
    #main()
    # Step 1: Create graph
    # A = mmread("/workspaces/flow-shop-scheduling/tests/dendrimer_matrix.mtx")
    # Graph = nx.convert_node_labels_to_integers(nx.Graph(A))
    # Graph.remove_edges_from(nx.selfloop_edges(Graph))
    # G=Graph
    k_list=[2,4,6]
    G = nx.erdos_renyi_graph(n=20, p=0.1, seed=42)
    n = G.number_of_nodes()

    
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Original Graph (Unpartitioned)")
    plt.savefig("original_graph.png", format='png', dpi=300)
    plt.close()
    print("Saved: original_graph.png")

    for k in k_list:

        
        user_alpha = 2.0
        user_beta = 1.625
        user_anneal_time = 30
        user_num_reads = 100

        
        bqm = build_k_concurrent_qubo(G, k, beta=user_beta, alpha=user_alpha, gamma=5.0)

        from dwave.system import EmbeddingComposite, DWaveSampler
        sampler = EmbeddingComposite(DWaveSampler())
        sample = sampler.sample(
            bqm,
            annealing_time=user_anneal_time,
            num_reads=user_num_reads
        ).first.sample

        
        partitions, violations = extract_partitions_with_violations(sample, n, k)
        print(f"Number of cut edges for {k} partitions is : {compute_cut_edges(G,partitions)}")
        print(f"{violations}")
        draw_partitioned_graph(G, partitions, filename="partitioned_graph_user_input.png")
        print("Saved: partitioned_graph_user_input.png")

    

    
