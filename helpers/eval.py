import networkx as nx

from preprocessing.input import read_partitioned_graph


def calculate_edge_cut(nx_graph, partition_assignments):
    edge_cut = 0
    for node in nx_graph.nodes:
        for neighbor in nx_graph.neighbors(node):
            if partition_assignments[node - 1] != partition_assignments[neighbor - 1]:  # Node IDs start at 1
                edge_cut += 1
    return int(edge_cut / 2)


if __name__ == '__main__':
    nx_graph = read_partitioned_graph("../data/experimental/road/road-euroroad.graph", "../data/experimental/road/road-euroroad.16.0.ptn", 16)
    partitions = list(nx.get_node_attributes(nx_graph, "partition").values())
    assert nx_graph.graph["edge_cut"] == calculate_edge_cut(nx_graph, partitions)
    print(f"Edge cut: {nx_graph.graph['edge_cut']}")
