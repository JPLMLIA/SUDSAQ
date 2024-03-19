#!/usr/bin/env python3
#
# Steven Lu
# January 11, 2024

import os
import sys
import datetime
import numpy as np
import xarray as xr
from region import Dataset
from region import MOMO_V4_VARS_SEL
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from networkx import Graph
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
import itertools
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


# Variables for making clusters
CLUSTER_VARS = [
    'momo.no',
    'momo.no2',
    'momo.co',
    'momo.ch2o',
    'momo.u',
    'momo.v',
    'momo.t',
    'momo.ps',
    'momo.2dsfc.NH3',
    'momo.2dsfc.BrOX',
    'momo.hno3',
    'momo.slrc',
    'momo.2dsfc.HO2',
    'momo.2dsfc.C2H6',
    'momo.2dsfc.C5H8'
]


class ClusterBipartiteGraph(Graph):
    def __init__(self, source_centers, target_centers):
        super(ClusterBipartiteGraph, self).__init__()

        # Create bipartite graph
        self.create(source_centers, target_centers)

    def create(self, source_centers, target_centers):
        # Add source cluster centers to partition x
        for ind, cc in enumerate(source_centers):
            self._add_node_to_partition_x(ind, cc)

        # Add target cluster centers to partition y
        for ind, cc in enumerate(target_centers):
            self._add_node_to_partition_y(ind, cc)

        # Get all nodes from partitions x and y
        node_partition_x = self._get_nodes_from_partition_x()
        node_partition_y = self._get_nodes_from_partition_y()

        # Calculate pair-wise distance
        pairs = itertools.product(node_partition_x, node_partition_y)
        for node_x, node_y in pairs:
            dist = distance.cosine(node_x[1]['cluster_center'],
                                   node_y[1]['cluster_center'])
            self.add_edge(node_x[0], node_y[0], distance=dist)

    def get_cost_matrix(self):
        # Get all nodes from partitions x and y
        nodenames_partition_x = self._get_nodenames_from_partition_x()
        nodenames_partition_y = self._get_nodenames_from_partition_y()

        # Get the cost matrix of the bipartite graph
        cost_matrix = biadjacency_matrix(self,
                                         nodenames_partition_x,
                                         nodenames_partition_y,
                                         weight='distance')

        return cost_matrix

    def _add_node_to_partition_x(self, node_id, cluster_center):
        self.add_node(f'x{node_id}', cluster_center=cluster_center, partition=0)

    def _add_node_to_partition_y(self, node_id, cluster_center):
        self.add_node(f'y{node_id}', cluster_center=cluster_center, partition=1)

    def _add_edge(self, node_x, node_y, dist):
        self.add_edge(node_x, node_y, distance=dist)

    def _get_nodes_from_partition_x(self):
        return [n for n in self.nodes(data=True) if n[1]['partition'] == 0]

    def _get_nodenames_from_partition_x(self):
        return [n for n, a in self.nodes(data=True) if a['partition'] == 0]

    def _get_nodes_from_partition_y(self):
        return [n for n in self.nodes(data=True) if n[1]['partition'] == 1]

    def _get_nodenames_from_partition_y(self):
        return [n for n, a in self.nodes(data=True) if a['partition'] == 1]


def main(momo_files, toar_mask_file, n_clusters, out_dir, feature_set):
    for f in momo_files:
        if not os.path.exists(f):
            print(f'Input file does not exist: {f}')
            sys.exit(1)

    # Load in MOMO files
    ds = xr.open_mfdataset(momo_files, engine='netcdf4', lock=False, parallel=True)
    ds = Dataset(ds)
    if feature_set == '15vars':
        ds = ds[CLUSTER_VARS]
    elif feature_set == 'momo_v4':
        ds = ds[MOMO_V4_VARS_SEL]
    ds = ds.sortby('lon')

    # Convert 2-hourly to daily data
    time = ds.time.dt.time
    time_mask = (datetime.time(0) <= time) & (time <= datetime.time(23))
    ds = ds.where(time_mask, drop=True)
    ds.coords['time'] = ds.time.dt.floor('1d')
    ds = ds.groupby('time').mean()

    # Convert dataset to ML-ready format
    data = ds.to_array()
    data = data.stack({'loc': ['lon', 'lat']}).transpose('time', 'loc', 'variable')
    loc = data.indexes['loc'].values
    loc = np.array(list(map(np.array, loc)))

    # Load in TOAR-2 binary mask file
    toar_mask = np.load(toar_mask_file)
    toar_mask = toar_mask.flatten(order='F')

    # Main loop
    cluster_labels = np.zeros((len(data), len(data[0])), dtype=np.int32)
    cluster_centers = np.zeros((2, n_clusters, len(CLUSTER_VARS)), dtype=np.float32)
    for day_ind, day in enumerate(range(len(data))):
        daily_data = data[day, :, :]

        # Normalization
        scaler = StandardScaler()
        daily_data = scaler.fit_transform(daily_data)

        # Clustering
        mbk = MiniBatchKMeans(
            init='k-means++', n_clusters=n_clusters, batch_size=100, n_init=10,
            max_no_improvement=10, verbose=0, random_state=398)
        labels = mbk.fit_predict(daily_data)
        cluster_labels[day, :] = labels
        if day_ind == 0:
            cluster_centers[0] = mbk.cluster_centers_
        else:
            cluster_centers[1] = mbk.cluster_centers_

            # Create bipartite graph
            bigraph = ClusterBipartiteGraph(cluster_centers[0],
                                            cluster_centers[1])

            # Get the cost matrix of the bipartite graph
            cost_matrix = bigraph.get_cost_matrix().toarray()

            # Solve the optimal assignment problem by minimizing the cost
            # of assigning cluster 2's labels to cluster 1's labels
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Reassign cluster labels
            for r, c in zip(row_ind, col_ind):
                labels[labels == c] = r

        # Generate daily cluster map
        cluster_map_file = os.path.join(out_dir, f'cluster_map_k{n_clusters}_day{day}.png')
        fig, ax = plt.subplots(figsize=(24, 18),
                               subplot_kw={'projection': ccrs.PlateCarree()})
        ax.add_feature(cf.COASTLINE)
        ax.add_feature(cf.BORDERS)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        gl.xlabel_style = {'size': 15, 'color': 'gray'}
        colors = iter(cm.tab20(np.linspace(0, 1, n_clusters)))
        for ind, k in enumerate(range(n_clusters)):
            in_group = labels == k
            ax.scatter(loc[in_group, 0], loc[in_group, 1], c=next(colors),
                       label=f'Cluster {ind}')

        ax.scatter(loc[:, 0], loc[:, 1], toar_mask.astype(int), c='black',
                   label='TOAR-2 stations')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
                  shadow=True, ncol=5, fontsize=20)
        plt.title(str(ds.indexes['time'][day_ind]).split(' ')[0])
        plt.savefig(cluster_map_file)
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--momo_files', nargs='+', required=True)
    parser.add_argument('--toar_mask_file', type=str, required=True)
    parser.add_argument('--n_clusters', type=int, default=15)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--feature_set', type=str, choices=['15vars', 'momo_v4'])

    args = parser.parse_args()
    main(**vars(args))
