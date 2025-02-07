from sklearn.neighbors import NearestNeighbors
import numpy as np

# def analyze_latent_knn(latent_vectors, k=5):
#     # Reshape if needed - each row should be one latent vector
#     X = np.array(latent_vectors)
#     if len(X.shape) == 1:
#         X = X.reshape(-1, 1)
        
#     # Create and fit KNN model
#     knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
#     knn.fit(X)
    
#     # Find neighbors for each point
#     distances, indices = knn.kneighbors(X)
    
#     return distances, indices

def analyze_latent_knn(latent_vectors, k=5):
    # Flatten each 2x2 matrix into a vector of length 4
    array = np.array(latent_vectors)
    X = np.array(latent_vectors).reshape(len(latent_vectors), -1)
    
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X)
    
    return knn.kneighbors(X)

# Usage:
# latent_vectors = [
#     [[0.8, 0.3], [-0.2, 0.7]],  # Cluster 1 (positive values)
#     [[0.7, 0.4], [-0.1, 0.6]],
#     [[0.9, 0.2], [-0.3, 0.8]],
#     [[0.75, 0.35], [-0.15, 0.65]],
#     [[0.85, 0.25], [-0.25, 0.75]],
#     [[0.65, 0.45], [-0.05, 0.55]],
    
#     [[-0.8, -0.3], [0.2, -0.7]], # Cluster 2 (negative values)
#     [[-0.7, -0.4], [0.1, -0.6]],
#     [[-0.9, -0.2], [0.3, -0.8]],
#     [[-0.75, -0.35], [0.15, -0.65]],
#     [[-0.85, -0.25], [0.25, -0.75]],
#     [[-0.65, -0.45], [0.05, -0.55]],
    
#     [[0.1, 0.1], [0.1, 0.1]],   # Cluster 3 (small values)
#     [[0.15, 0.15], [0.15, 0.15]],
#     [[0.05, 0.05], [0.05, 0.05]],
#     [[0.12, 0.12], [0.12, 0.12]],
#     [[0.08, 0.08], [0.08, 0.08]],
#     [[0.03, 0.03], [0.03, 0.03]],
    
#     [[1.5, -1.5], [-1.5, 1.5]], # Cluster 4 (extreme values)
#     [[1.4, -1.4], [-1.4, 1.4]],
#     [[1.6, -1.6], [-1.6, 1.6]],
#     [[1.45, -1.45], [-1.45, 1.45]],
#     [[1.55, -1.55], [-1.55, 1.55]],
#     [[1.35, -1.35], [-1.35, 1.35]]
# ]

latent_vectors = [
   # Cluster 1 - Positive values around 0.5
   [0.45, 0.52, 0.48, 0.51, 0.47, 0.53, 0.49, 0.50],
   [0.46, 0.51, 0.49, 0.52, 0.48, 0.54, 0.47, 0.51],
   [0.44, 0.53, 0.47, 0.50, 0.46, 0.52, 0.48, 0.49],
   [0.47, 0.50, 0.46, 0.49, 0.45, 0.51, 0.50, 0.48],
   [0.48, 0.49, 0.45, 0.48, 0.44, 0.50, 0.51, 0.47],
   [0.49, 0.48, 0.44, 0.47, 0.43, 0.49, 0.52, 0.46],

   # Cluster 2 - Negative values around -0.5
   [-0.45, -0.52, -0.48, -0.51, -0.47, -0.53, -0.49, -0.50],
   [-0.46, -0.51, -0.49, -0.52, -0.48, -0.54, -0.47, -0.51],
   [-0.44, -0.53, -0.47, -0.50, -0.46, -0.52, -0.48, -0.49],
   [-0.47, -0.50, -0.46, -0.49, -0.45, -0.51, -0.50, -0.48],
   [-0.48, -0.49, -0.45, -0.48, -0.44, -0.50, -0.51, -0.47],
   [-0.49, -0.48, -0.44, -0.47, -0.43, -0.49, -0.52, -0.46],

   # Cluster 3 - Small values around 0.1
   [0.09, 0.11, 0.08, 0.12, 0.10, 0.09, 0.11, 0.10],
   [0.10, 0.12, 0.09, 0.11, 0.09, 0.10, 0.12, 0.11],
   [0.11, 0.10, 0.10, 0.10, 0.08, 0.11, 0.10, 0.12],
   [0.08, 0.09, 0.11, 0.09, 0.11, 0.12, 0.09, 0.10],
   [0.12, 0.08, 0.12, 0.08, 0.12, 0.10, 0.08, 0.09],
   [0.10, 0.11, 0.10, 0.11, 0.10, 0.09, 0.11, 0.08],

   # Cluster 4 - Large alternating values
   [1.5, -1.5, 1.5, -1.5, 1.5, -1.5, 1.5, -1.5],
   [1.4, -1.4, 1.4, -1.4, 1.4, -1.4, 1.4, -1.4],
   [1.6, -1.6, 1.6, -1.6, 1.6, -1.6, 1.6, -1.6],
   [1.45, -1.45, 1.45, -1.45, 1.45, -1.45, 1.45, -1.45],
   [1.55, -1.55, 1.55, -1.55, 1.55, -1.55, 1.55, -1.55],
   [1.35, -1.35, 1.35, -1.35, 1.35, -1.35, 1.35, -1.35]
]


if __name__ == "__main__":
    distances, indices = analyze_latent_knn(latent_vectors)
