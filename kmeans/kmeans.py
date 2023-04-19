import numpy as np

class KMeans:
    def __init__(self, n_clusters, cluster_dist_tol=0.1, max_iter=100) -> None:
        self.n_clusters = n_clusters
        self.cluster_dist_tolerance = cluster_dist_tol
        self.max_iter = max_iter

    def fit(self, data):
        self.data = data
        self.data_dim = data.shape[1]
        self.data_size = data.shape[0]

        self.cluster_centroids, self.labels = self.iterate_cluster_centroid_updates()

    def predict(self, data):
        assert data.shape[0] == self.data_size
        assert data.shape[1] == self.data_dim

        predict_data_size = data.shape[0]

        _, labels = self.get_point_cluster_map_labels(
                                        self.cluster_centroids,
                                        data_size=predict_data_size
                                        )
        
        return labels
    

    def iterate_cluster_centroid_updates(self):

        iter = 0
        cluster_center_dist = float('inf')
        self.cluster_distances = list()
        cluster_centroids = self.get_cluster_centers_start()

        # continue until the mean cluster distance is lower than the tolerance or the maximum iteration is reached
        while cluster_center_dist >= self.cluster_dist_tolerance and iter < self.max_iter:

            iter += 1
            
            point_cluster_map, labels = self.get_point_cluster_map_labels(cluster_centroids, 
                                                                          self.data_size)
            updated_centroids = self.update_cluster_centers(point_cluster_map=point_cluster_map)
            cluster_center_dist = self.get_cluster_updated_cluster_center_distance(cluster_centers=cluster_centroids, 
                                                            updated_cluster_centers=updated_centroids)
            self.cluster_distances.append(cluster_center_dist)
            cluster_centroids = updated_centroids
        return cluster_centroids, labels

    
    def get_cluster_centers_start(self):
        # assign 3 points that will act as cluster centers
        return np.take(self.data, np.random.choice(self.data_size, self.n_clusters, replace=False), axis=0)   
    
    def get_point_cluster_map_labels(self, cluster_centers, data_size):

        labels = np.repeat(-1, data_size)
        # assign each data point to a cluster center

        point_cluster_map = {}
        for center in range(self.n_clusters):
            point_cluster_map[center] = list()

        for ipoint in range(data_size):
            # get point values
            point = self.data[ipoint,:]

            # empty list for distance of a point from all the clusters
            cluster_dist = list()
            for icluster_center in range(self.n_clusters):
                cluster_point = cluster_centers[icluster_center,:]
                # compute distance of the point from all the cluster centroids
                temp_dist = self.get_distance(cluster_point, point)
                cluster_dist.append(temp_dist)

            # get the cluster index of the cluster that the point is assigned to
            cluster_idx = np.argmin(cluster_dist)
            labels[ipoint] = cluster_idx
            
            # append each cluster to a cluster center
            point_cluster_map[cluster_idx].append(ipoint)
        return point_cluster_map, labels
    

    def update_cluster_centers(self, point_cluster_map):
        # Get new centroids
        # formula: cluster centers is just the mean of all the points in that cluster in the cartesian coordinate system
        updated_cluster_centers = np.zeros((self.n_clusters, self.data_dim))

        for icluster in range(self.n_clusters):
            ipoints_in_cluster = point_cluster_map[icluster]
            # filter points by cluster index
            points_in_cluster = np.take(self.data, ipoints_in_cluster, axis=0)
            updated_cluster_centers[icluster,:] = np.mean(points_in_cluster, axis=0)
        return updated_cluster_centers
    
    def get_cluster_updated_cluster_center_distance(self, cluster_centers, updated_cluster_centers):
        cluster_center_dist = 0
        for icluster in range(self.n_clusters):
            temp_cluster_dist = self.get_distance(cluster_centers[icluster,:], updated_cluster_centers[icluster,:])
            cluster_center_dist += temp_cluster_dist
        return cluster_center_dist

        
    @staticmethod
    def get_distance(point1, point2):
        diff = point1 - point2
        sum_sq = np.dot(diff.T, diff)
        return np.sqrt(sum_sq)

