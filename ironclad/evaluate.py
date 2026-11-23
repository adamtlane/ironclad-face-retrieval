import os
import time
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image

from modules.extraction.preprocessing import Preprocessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.index.lsh import FaissLSH
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.search import FaissSearch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GALLERY_DIR = "storage/gallery" # Directory obtaining location.
PROBE_DIR = "storage/probe"

INDEX_TYPE = "lsh" # Configuration option: "lsh", "hnsw", or "bruteforce". Determines index type.
METRIC = "cosine" # Configuration option: "cosine" or "euclidean". Determines distance metric.
K = 5 # Configuration option: 1, 3, or 5. Top-K results to retrieve.
MODEL_NAME = "vggface2" # Configuration option: `casia-webface` or `vggface2`. Pretrained embedding model name.

RESULTS_DIR = os.path.join("evaluation_results", MODEL_NAME, INDEX_TYPE, METRIC, f"k{K}") # Output directory for results and saved index.
INDEX_PATH = os.path.join(RESULTS_DIR, f"faiss_{INDEX_TYPE}_index.pkl")

def ensure_dir(path):
    """
    Ensure the parent directory of a given path exists.

    Parameters:
        path (str): Path to a file or directory.

    Returns:
        None
    """
    dir_path = path if os.path.isdir(path) else os.path.dirname(path)
    
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def load_index():
    """
    Load the saved index from disk if it exists.

    Returns:
        FaissIndex: Loaded index instance.

    Raises:
        RuntimeError: If the index file is not found.
    """
    if INDEX_TYPE == "hnsw": # Based on FAISS index type.
        index_class = FaissHNSW
    elif INDEX_TYPE == "lsh":
        index_class = FaissLSH
    else:
        index_class = FaissBruteForce

    if os.path.exists(INDEX_PATH): # Define index path for saving purposes, based on previous configuration option/choice.
        print(f"Loading index from {INDEX_PATH} ...")

        return index_class.load(INDEX_PATH)
    else:
        print(f"No saved index found at {INDEX_PATH}. Building from gallery ...")
        raise RuntimeError("Index not found. Building required.")

def build_gallery_index(preprocessor, model, index):
    """
    Create an index from all gallery images and store their embeddings.

    Parameters:
        preprocessor (Preprocessing): Image preprocessing instance.
        model (Embedding): Embedding model for feature extraction.
        index (FaissIndex): Empty index to populate.

    Returns:
        FaissIndex: Index populated with gallery embeddings.
    """
    print("Building gallery index ...")

    embeddings = []
    metadatas = []

    for person_name in os.listdir(GALLERY_DIR): # Based on gallery directory.
        person_dir = os.path.join(GALLERY_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            if img_file.startswith("._"):
                continue # Skip any macOS metadata files.
            img_path = os.path.join(person_dir, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocessor.process(img) # Perform preprocessing.
                emb = model.encode(img_tensor).reshape(-1) # Reshape tensor.
                embeddings.append(emb)
                metadatas.append(person_name)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    index.add_embeddings(embeddings, metadatas)
    print(f"Added {len(embeddings)} embeddings to the index.")

    return index

def evaluate():
    """
    Evaluate face retrieval performance on the probe set against the gallery.

    Performs preprocessing, embedding, nearest-neighbor retrieval,
    and computes top-K accuracy. Results are saved to JSON and CSV.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = Preprocessing(image_size=160)
    model = Embedding(pretrained=MODEL_NAME, device=device)

    try:
        index = load_index() # Build a new index if one doesn't exist.
    except RuntimeError:
        index_dim = 512
        if INDEX_TYPE == "hnsw":
            index = FaissHNSW(dim=index_dim, metric=METRIC)
        elif INDEX_TYPE == "lsh":
            index = FaissLSH(dim=index_dim)
        else:
            index = FaissBruteForce(dim=index_dim)

        index = build_gallery_index(preprocessor, model, index)
        
        ensure_dir(INDEX_PATH) # Save the newly created index.
        index.save(INDEX_PATH)

        print(f"Index saved to {INDEX_PATH}.")

    searcher = FaissSearch(index, metric=METRIC)
    results = []
    
    ensure_dir(RESULTS_DIR) # Ensure the output directory exists.

    for person_name in os.listdir(PROBE_DIR): # Process each probe image and evaluate retrieval accuracy.
        person_dir = os.path.join(PROBE_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for probe_file in os.listdir(person_dir):
            if probe_file.startswith("._"):
                continue
            probe_path = os.path.join(person_dir, probe_file)
            true_identity = person_name

            try:
                img = Image.open(probe_path).convert("RGB")
                img_tensor = preprocessor.process(img)
                start_time = time.time()
                emb = model.encode(img_tensor)
                distances, indices, metadatas = searcher.search(emb, K)
                elapsed = time.time() - start_time
                
                retrieved = [m for m in metadatas[0] if m is not None] # Extract non-null retrieved identities.
                top_k_correct = true_identity in retrieved[:K]

                results.append({
                    "probe_image": probe_file,
                    "true_identity": true_identity,
                    "retrieved": retrieved,
                    "distances": distances[0].tolist(),
                    "time_sec": elapsed,
                    "top_k_correct": top_k_correct,
                })

                print(f"Probe {probe_file}: top-{K} correct? {top_k_correct} (retrieved: {retrieved})")

            except Exception as e:
                print(f"Error processing probe {probe_file}: {e}")

    json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{INDEX_TYPE}_{METRIC}_k{K}.json") # Save the raw evaluation results.

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    total = len(results)
    correct = sum(r["top_k_correct"] for r in results)
    accuracy = correct / total if total > 0 else 0 # Compute the accuracy.

    print(f"\n=== Summary for {INDEX_TYPE.upper()} | Metric: {METRIC} | K: {K} ===")
    print(f"Total probes: {total}")
    print(f"Top-{K} accuracy: {accuracy*100:.2f}%")

    csv_path = os.path.join(RESULTS_DIR, f"evaluation_summary_{INDEX_TYPE}_{METRIC}_k{K}.csv") # Define the .csv file path.
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False) # Save summary to CSV.

    print(f"Results saved to {json_path} and {csv_path}")

if __name__ == "__main__":
    evaluate()