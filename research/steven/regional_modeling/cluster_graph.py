#!/usr/bin/env python3
#
# Steven Lu
# October 10, 2023

import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Compute Euclidean distance of two arrays
def distance(c1, c2):
    c = c1 - c2

    return np.sqrt(np.dot(c.T, c))


def main(cluster_npz_file, out_dir):
    if not os.path.exists(cluster_npz_file):
        print(f'Input file does not exist: {cluster_npz_file}')
        sys.exit(1)

    cluster_dict = dict(np.load(cluster_npz_file, allow_pickle=True))

    # Overall graph
    G = nx.Graph()
    cluster_keys = cluster_dict.keys()
    for k1 in cluster_keys:
        if not G.has_node(k1):
            G.add_node(k1, n_total=cluster_dict[k1].item()['n_total'],
                       n_toar=cluster_dict[k1].item()['n_toar'])

        for k2 in cluster_keys:
            if k1 == k2:
                continue

            c1 = cluster_dict[k1].item()['centroid']
            c2 = cluster_dict[k2].item()['centroid']
            d = distance(c1, c2)

            if not G.has_edge(k1, k2):
                G.add_edge(k1, k2, distance=f'{d:.2f}')

    with open(os.path.join(out_dir, 'cluster_graph.pickle'), 'wb') as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure(figsize=(20, 20))
    options = {
        "font_size": 10,
        "node_size": 1000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 2,
        "width": 2,
    }
    pos = nx.spring_layout(G, k=10 / np.sqrt(G.order()))
    nx.draw_networkx(G, pos, **options)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'distance'))
    plt.axis('off')
    plt.savefig(os.path.join(out_dir, 'cluster_graph.png'))
    plt.close()

    # Individual graphs
    for k1 in cluster_keys:
        G = nx.Graph()

        if not G.has_node(k1):
            G.add_node(k1, n_total=cluster_dict[k1].item()['n_total'],
                       n_toar=cluster_dict[k1].item()['n_toar'])

        for k2 in cluster_keys:
            if k1 == k2:
                continue

            c1 = cluster_dict[k1].item()['centroid']
            c2 = cluster_dict[k2].item()['centroid']
            d = distance(c1, c2)

            if not G.has_edge(k1, k2):
                G.add_edge(k1, k2, distance=f'{d:.2f}')

        plt.figure(figsize=(10, 10))
        pos = nx.spring_layout(G, k=10 / np.sqrt(G.order()))
        nx.draw_networkx(G, pos, **options)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'distance'))
        plt.axis('off')
        plt.savefig(os.path.join(out_dir, f'{k1}_graph.png'))
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('cluster_npz_file', type=str)
    parser.add_argument('out_dir', type=str)

    args = parser.parse_args()
    main(**vars(args))
