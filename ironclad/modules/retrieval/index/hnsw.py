import faiss
import numpy as np
import pickle

########################################
# ASSIGNMENT 5, TASK 3: 
#         Implement `/add` endpoint to
#         add the provided image to the 
#         catalog.
########################################

class FaissHNSW:
    """
    A FAISS HNSW index for storing embeddings and their associated metadata using
    the Hierarchical Navigable Small World (HNSW) algorithm.

    This class leverages FAISS's HNSW index for fast approximate nearest neighbor search
    in large datasets. It does not require a training step and provides efficient insertion
    and retrieval of embeddings based on similarity.

    Attributes:
        dim (int): Dimensionality of the embeddings.
        metadata (list): Metadata associated with each embedding.
        index (faiss.IndexHNSWFlat): FAISS index object using HNSW for ANN search.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes a FaissHNSW instance with configurable HNSW parameters.

        Parameters:
            dim (int): Dimensionality of the embeddings.
            **kwargs:
                - M (int): Number of neighbors per node in the HNSW graph (default=32).
                - efConstruction (int): Candidate pool size during graph construction (default=40).
                - metric (str): Distance metric to use; either 'euclidean', 'cosine', or 'dot_product'.
        """
        self.dim = dim
        self.metadata = []
        self.metric = kwargs.get('metric', 'euclidean').lower()
        self.m = kwargs.get('M', 32)
        self.efConstruction = kwargs.get('efConstruction', 40)

        if self.metric in ['euclidean', 'minkowski']: # Initialize HNSW index with specified metric.
            self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_L2)
        elif self.metric in ['cosine', 'dot_product']:
            self.index = faiss.IndexHNSWFlat(dim, self.m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'cosine', or 'dot_product'.") # Raise error if metric is not present. 

        self.index.hnsw.efConstruction = self.efConstruction

    def add_embeddings(self, new_embeddings, new_metadata):
        """
        Adds embeddings and associated metadata to the HNSW index.

        Parameters:
            new_embeddings (list or np.ndarray): Embeddings to add, each of shape (dim,).
            new_metadata (list): Metadata corresponding to each embedding.

        Raises:
            ValueError: If lengths of embeddings and metadata do not match.
            ValueError: If any embedding does not match the expected dimensionality.
        """
        if len(new_embeddings) != len(new_metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        for emb, meta in zip(new_embeddings, new_metadata):
            emb = np.array(emb) # Convert embedding to NumPy array.

            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")
            
            self.metadata.append(meta)
            vector = emb.astype(np.float32).reshape(1, -1)

            if self.metric == 'cosine': # Normalize if cosine similarity is used.
                faiss.normalize_L2(vector)

            self.index.add(vector)

    def get_metadata(self, idx):
        """
        Retrieves metadata for a given embedding index.

        Parameters:
            idx (int): Index of the embedding.

        Returns:
            Any: The metadata associated with the embedding.

        Raises:
            IndexError: If index is out of bounds.
        """
        if idx < 0 or idx >= len(self.metadata): # Out of bounds error return.
            raise IndexError("Index out of bounds.")
        
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the current index and metadata to disk.

        Parameters:
            filepath (str): Path to the output pickle file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a FaissHNSW instance from disk.

        Parameters:
            filepath (str): Path to the pickle file.

        Returns:
            FaissHNSW: Restored index instance.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)

        return instance

    def search(self, query, k):
        """
        Searches for the k nearest neighbors of a query embedding.

        Parameters:
            query (list or np.ndarray): Query embedding of shape (dim,).
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            tuple:
                - distances (np.ndarray): Distances to nearest neighbors.
                - indices (np.ndarray): Indices of nearest neighbors.
        """
        query = np.array(query, dtype=np.float32).reshape(1, -1)

        if self.metric == 'cosine':
            faiss.normalize_L2(query)

        return self.index.search(query, k)

if __name__ == "__main__": # Example usage for testing.
    
    index = FaissHNSW(dim=4, M=16, efConstruction=50) # Initialize HNSW index with 4D embeddings and custom HNSW settings.

    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ] # Dummy embeddings and labels.

    identity_metadata = ["Alice", "Bob", "Charlie"]
    index.add_embeddings(embeddings, identity_metadata) # Add to index.
    
    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32) # Query the index.
    k = 2
    distances, indices = index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]]

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)
    
    filepath = "faiss_hnsw_index.pkl" 
    index.save(filepath) # Save to disk.

    print(f"Index saved to {filepath}.")

    loaded_index = FaissHNSW.load(filepath) # Load from disk and verify.

    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0)) # Display the printed results.