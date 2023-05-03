#!/usr/bin/env python3
# Determine the optimal number of clusters with the Elbow method
# https://www.analyticsvidhya.com/blog/2021/05/k-mean-getting-the-optimal-number-of-clusters/
#
# Steven Lu
# May 2, 2023

import numpy as np
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt


def main(pca_npy_file, out_plot):
    data_arr = np.load(pca_npy_file)
    sqrt_dist_sum = []
    K = range(1, 10)

    for n_clusters in K:
        mbk = MiniBatchKMeans(
            init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
            max_no_improvement=10, verbose=0)
        mbk.fit(data_arr)
        sqrt_dist_sum.append(mbk.inertia_)

    plt.plot(K, sqrt_dist_sum, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sum of squared distances')
    plt.savefig(out_plot)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('pca_npy_file', type=str)
    parser.add_argument('out_plot', type=str)

    args = parser.parse_args()
    main(**vars(args))
