import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import NearestNeighbors

#Distance matrix from numpy (dummy)
precomputed_distances = np.random.random((6,6)) 
precomputed_distances = precomputed_distances.T @ precomputed_distances
precomputed_distances[range(6), range(6)] = 0
K = 3
graph = kneighbors_graph(precomputed_distances, K, metric='precomputed', mode='connectivity', include_self=False)
adjacency_matrix = graph.toarray()

#Get top 5 neighbours from precomputed distance matrix
nn = NearestNeighbors(n_neighbors=K, metric='precomputed')
nn.fit(precomputed_distances)

#Fetch kneighbors
distances, indexes = nn.kneighbors()


print(indexes)
print('')
print(adjacency_matrix)
print('')
print(precomputed_distances)
