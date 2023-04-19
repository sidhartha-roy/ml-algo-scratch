# Kmeans
Kmeans is an unsupervised clustering algorithm, that divides groups of points into clusters

K is the predefined number of clusters to be formed by the algorithm.

## Implementation of the kmeans algorithm

step 1: select the value of k to decide the number of clusters (n_clusters) to be formed

step 2: select k random points that will act as cluster centroids

step 3: assign each point in the data to a cluster centroid, by computing the distance between the cluster centroid and the data point

step 4: now you have a set of clusters, for each of the new clusters obtain the new cluster centroids. This can be computed by 
        obtaining the mean of the distance of all the points in each cluster.

step 5: Next compute the distance between the old cluster centroid and updated cluster centroid. this is required to know how long the clustering algorithm will run. I can set a tolerance level based on the distance between the new cluster centroid and the old one. And if they are about the same then there is not much point running the algorithm longer.

step 6: repeat step 3 to step 5 until the the mean cluster distance (updated vs old cluster distances) is lower than the tolerance or the max iteration is reached.