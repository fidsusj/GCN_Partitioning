import time

import networkx as nx


def read_partitioned_graph(graph_path, partition_path, k):
    print("Reading input data ...")
    start = time.time()

    nx_graph = nx.empty_graph()
    edge_cut = 0

    with open(partition_path) as partition:
        for index, line in enumerate(partition):
            partition = int(line.strip())
            nx_graph.add_node(index + 1,  # Node IDs start at 1
                              weight=1,
                              partition=partition)

    with open(graph_path) as graph:
        for index, line in enumerate(graph):
            if index == 0:  # First line contains #nodes and #edges
                continue
            for edge_target in line.split():
                edge_target = int(edge_target)
                nx_graph.add_edge(index, edge_target, weight=1)
                if nx_graph.nodes[index]["partition"] != nx_graph.nodes[edge_target]["partition"]:
                    edge_cut += 1

    nx_graph.graph.update({"edge_cut": int(edge_cut / 2)})

    end = time.time()
    print(f"io time in sec: {end - start}")
    return nx_graph


if __name__ == '__main__':
    uk = read_partitioned_graph("../data/experimental/road/road-euroroad.graph", "../data/experimental/road/road-euroroad.16.0.ptn", 16)
    print(f"Nodes: {uk.number_of_nodes()}, Edges: {uk.number_of_edges()}")
