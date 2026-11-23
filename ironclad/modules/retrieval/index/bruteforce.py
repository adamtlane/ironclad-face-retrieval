import numpy as np
import pickle
import faiss

########################################
# ASSIGNMENT 5, TASK 1: 
#         Implement `/add` endpoint to
#         add the provided image to the 
#         catalog.
########################################

class FaissBruteForce:
    """
    A brute-force FAISS index for storing and retrieving image embeddings with metadata.

    This class supports similarity search using Euclidean, Cosine, or Dot Product distances.

    Attributes:
        dim (int): Dimensionality of each embedding vector.
        metadata (list): List to store metadata associated with embeddings.
        metric (str): Distance metric to use ('euclidean', 'cosine', or 'dot_product').
        index (faiss.IndexFlat): FAISS index object used for efficient search.
    """

    def __init__(self, dim, metric='euclidean'):
        """
        Initialize the FAISS brute-force index.

        Parameters:
            dim (int): The dimension of each embedding vector.
            metric (str): Similarity metric; one of 'euclidean', 'cosine', or 'dot_product'.

        Raises:
            ValueError: If an unsupported distance metric is provided.
        """
        self.dim = dim
        self.metadata = [] # Stores metadata for each embedding.
        self.metric = metric.lower()

        if self.metric in ['euclidean', 'minkowski']:
            self.index = faiss.IndexFlatL2(dim) # Recall that L2 is Euclidean.
        elif self.metric in ['cosine', 'dot_product']:
            self.index = faiss.IndexFlatIP(dim) # Cosine similarity is implemented via inner product with normalized vectors.
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.")

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and associated metadata to the index.

        Parameters:
            embeddings (list or np.ndarray): Embedding vectors of shape (N, dim).
            metadata (list): A list of metadata (e.g. names) of length N.

        Raises:
            ValueError: If metadata length doesn't match embeddings, or dimensions mismatch.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        for emb, meta in zip(embeddings, metadata):
            emb = np.array(emb)

            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")

            self.metadata.append(meta)
            vector = emb.astype(np.float32).reshape(1, -1)

            if self.metric == 'cosine':
                faiss.normalize_L2(vector) # Normalize vectors for cosine similarity.

            self.index.add(vector)

    def get_metadata(self, idx):
        """
        Retrieve metadata by embedding index.

        Parameters:
            idx (int): The index of the embedding.

        Returns:
            object: Metadata associated with the embedding.
        """
        if idx < 0 or idx >= len(self.metadata): # Out of bounds error return.
            raise IndexError("Index out of bounds.")
        
        return self.metadata[idx]

    def save(self, filepath):
        """
        Save the current index and metadata to disk.

        Parameters:
            filepath (str): Path to the file where the index will be saved.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a saved FaissBruteForce instance from disk.

        Parameters:
            filepath (str): Path to the file from which to load.

        Returns:
            FaissBruteForce: Loaded instance.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)
            
        return instance

if __name__ == "__main__": # Example usage for testing.
    
    metric = 'cosine'
    index = FaissBruteForce(dim=4, metric=metric) # Example use-case for demonstration and testing.
    
    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ] # Dummy embeddings and labels.

    identity_metadata = ["Alice", "Bob", "Charlie"]
    index.add_embeddings(embeddings, identity_metadata) # Add to the FAISS index.

    query = [0.1, 0.2, 0.3, 0.4]
    k = 2
    query_vector = np.array(query).astype(np.float32).reshape(1, -1) # Perform the search functionality.

    if metric == 'cosine':
        faiss.normalize_L2(query_vector)

    distances, indices = index.index.search(query_vector, k) # Find the distances and indices via indexing the query vector.
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    filepath = "faiss_bruteforce_index.pkl"
    index.save(filepath) # Save and reload the index.

    print(f"Index saved to {filepath}.")

    loaded_index = FaissBruteForce.load(filepath)

    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0)) # Display the printed results.