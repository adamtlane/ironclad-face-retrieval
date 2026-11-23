import numpy as np
import pickle
import faiss

########################################
# ASSIGNMENT 5, TASK 2: 
#         Implement `/add` endpoint to
#         add the provided image to the 
#         catalog.
########################################

class FaissLSH:
    """
    An LSH-based FAISS index for storing embeddings and their associated metadata.

    This class uses Locality-Sensitive Hashing (LSH) to create an approximate nearest neighbor
    index, which is especially effective for high-dimensional data. It's designed to trade
    off some accuracy for significantly faster search times on large datasets.

    Attributes:
        dim (int): Dimensionality of the embeddings.
        metadata (list): Stores metadata associated with each added embedding.
        index (faiss.IndexLSH): FAISS LSH index for approximate nearest neighbor search.
    """

    def __init__(self, dim, **kwargs):
        """
        Initializes the FaissLSH index.

        Parameters:
            dim (int): Dimensionality of the embeddings.
            **kwargs: Optional parameters:
                - nbits (int): Number of bits used for hashing in the LSH index.
                               Defaults to 128 if not specified.
        """
        self.dim = dim
        self.metadata = []
        self.nbits = kwargs.get('nbits', 128)
        self.index = faiss.IndexLSH(dim, self.nbits) # Create a FAISS LSH index using the specified number of bits.

    def add_embeddings(self, embeddings, metadata):
        """
        Adds embeddings and their associated metadata to the FAISS index.

        Parameters:
            embeddings (list | np.ndarray): List or array of embeddings, each of length `dim`.
            metadata (list): List of metadata corresponding to each embedding.
        """
        if len(embeddings) != len(metadata):
            raise ValueError("The number of embeddings must match the number of metadata entries.")

        for emb, meta in zip(embeddings, metadata):
            emb = np.array(emb)

            if emb.shape[0] != self.dim:
                raise ValueError(f"Embedding has dimension {emb.shape[0]}, expected {self.dim}.")

            self.metadata.append(meta)
            vector = emb.astype(np.float32).reshape(1, -1) # Convert to float32 format and reshape for FAISS.
            self.index.add(vector) # Add vector to the FAISS index.

    def get_metadata(self, idx):
        """
        Retrieves metadata for a given index.

        Parameters:
            idx (int): Index of the embedding.

        Returns:
            Any: Metadata corresponding to the index.
        """
        if idx < 0 or idx >= len(self.metadata): # Out of bounds error return.
            raise IndexError("Index out of bounds.")
        
        return self.metadata[idx]

    def save(self, filepath):
        """
        Saves the FaissLSH instance (index + metadata) to a file using pickle.

        Parameters:
            filepath (str): Path to the output `.pkl` file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Loads a previously saved FaissLSH instance from a pickle file.

        Parameters:
            filepath (str): Path to the `.pkl` file.

        Returns:
            FaissLSH: Restored FaissLSH object.
        """
        with open(filepath, 'rb') as f:
            instance = pickle.load(f)

        return instance

    def search(self, query, k):
        """
        Searches the FAISS index for the `k` nearest neighbors of a query vector.

        Parameters:
            query (list | np.ndarray): The query embedding vector.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            tuple:
                distances (np.ndarray): Distances to the nearest neighbors.
                indices (np.ndarray): Indices of the nearest neighbors.
        """
        query = np.array(query, dtype=np.float32).reshape(1, -1)

        return self.index.search(query, k)

if __name__ == "__main__": # Example usage for testing.

    index = FaissLSH(dim=4, nbits=256) # Initialize index with 4D embeddings and 256-bit LSH hashing.

    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ] # Sample 4D embeddings and their corresponding names as metadata.

    identity_metadata = ["Alice", "Bob", "Charlie"]
    index.add_embeddings(embeddings, identity_metadata) # Add the embeddings and associated metadata.

    query = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32) # Define a query vector and search for the 2 closest neighbors.
    k = 2
    distances, indices = index.search(query, k)
    meta_results = [index.get_metadata(int(i)) for i in indices[0]] # Retrieve metadata associated with the found indices.

    print("Query Vector:", query)
    print("Distances:", distances)
    print("Indices:", indices)
    print("Metadata Results:", meta_results)

    filepath = "faiss_lsh_index.pkl"
    index.save(filepath) # Save the index to a pickle file.

    print(f"Index saved to {filepath}.")

    loaded_index = FaissLSH.load(filepath) # Load index back from file.

    print("Loaded Metadata for index 0:", loaded_index.get_metadata(0)) # Display the printed results.