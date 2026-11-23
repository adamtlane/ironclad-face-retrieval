import os
import time
import json
import torch
import numpy as np
import pandas as pd
import random
from PIL import Image

from modules.extraction.preprocessing import Preprocessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.index.lsh import FaissLSH
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.search import FaissSearch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GALLERY_DIR = "storage/gallery"
PROBE_DIR = "storage/probe"

INDEX_TYPE = "lsh"  # Configuration option: "lsh", "hnsw", or "bruteforce". Determines index type.
METRIC = "cosine"  # Configuration option: "cosine" or "euclidean". Determines distance metric.
K = 3  # Configuration option: 1, 3, or 5. Top-K results to retrieve.
MODEL_NAME = "casia-webface"  # Configuration option: `casia-webface` or `vggface2`. Pretrained embedding model name.

RESULTS_DIR = os.path.join("evaluation_results_m", MODEL_NAME, INDEX_TYPE, METRIC, f"k{K}")
INDEX_PATH = os.path.join(RESULTS_DIR, f"faiss_{INDEX_TYPE}_index.pkl")


def ensure_dir(dir_path):
    """
    Ensure the directory exists; if not, create it.

    Args:
        dir_path (str): Directory path to ensure.

    Returns:
        None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def load_index():
    """
    Load the saved index from disk if it exists.

    Returns:
        FaissIndex: Loaded index instance.

    Raises:
        RuntimeError: If the index file is not found.
    """
    if INDEX_TYPE == "hnsw":
        index_class = FaissHNSW
    elif INDEX_TYPE == "lsh":
        index_class = FaissLSH
    else:
        index_class = FaissBruteForce

    if os.path.exists(INDEX_PATH):
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

    for person_name in os.listdir(GALLERY_DIR):
        person_dir = os.path.join(GALLERY_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        for img_file in os.listdir(person_dir):
            if img_file.startswith("._"):
                continue
            img_path = os.path.join(person_dir, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocessor.process(img)
                emb = model.encode(img_tensor).reshape(-1)
                embeddings.append(emb)
                metadatas.append(person_name)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    index.add_embeddings(embeddings, metadatas)
    print(f"Added {len(embeddings)} embeddings to the index.")

    return index


def build_gallery_subset(preprocessor, model, index, m):
    """
    For each identity in the gallery, sample `m` images and add them to the index.

    Parameters:
        preprocessor (Preprocessing): Image preprocessing instance.
        model (Embedding): Embedding model for feature extraction.
        index (FaissIndex): Empty index to populate.
        m (int): Number of images per identity to sample.

    Returns:
        FaissIndex: Index populated with sampled gallery embeddings.
    """
    print(f"Building gallery with m={m} images per identity ...")

    embeddings = []
    metadatas = []

    for person_name in os.listdir(GALLERY_DIR):
        person_dir = os.path.join(GALLERY_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        img_files = [f for f in os.listdir(person_dir) if not f.startswith("._")]
        random.shuffle(img_files)
        selected_files = img_files[:m]

        for img_file in selected_files:
            img_path = os.path.join(person_dir, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = preprocessor.process(img)
                emb = model.encode(img_tensor).reshape(-1)
                embeddings.append(emb)
                metadatas.append(person_name)
            except Exception as e:
                print(f"Skipping {img_path} due to error: {e}")

    index.add_embeddings(embeddings, metadatas)
    print(f"Added {len(embeddings)} embeddings for m={m}.")

    return index


def evaluate(m_values=None):
    """
    Evaluate face retrieval performance.

    If m_values is None or empty, evaluate on full gallery (existing behavior).
    If m_values is a list of ints, evaluate on gallery subsets with m images per identity.

    Parameters:
        m_values (list[int] or None): List of image counts per identity to evaluate.

    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = Preprocessing(image_size=160)
    model = Embedding(pretrained=MODEL_NAME, device=device)

    index_dim = 512
    if INDEX_TYPE == "hnsw":
        index_class = FaissHNSW
    elif INDEX_TYPE == "lsh":
        index_class = FaissLSH
    else:
        index_class = FaissBruteForce

    if not m_values:
        # === Original full gallery evaluation ===
        try:
            index = load_index()
        except RuntimeError:
            index = index_class(dim=index_dim, metric=METRIC)
            index = build_gallery_index(preprocessor, model, index)
            ensure_dir(os.path.dirname(INDEX_PATH))
            index.save(INDEX_PATH)
            print(f"Index saved to {INDEX_PATH}.")

        searcher = FaissSearch(index, metric=METRIC)
        results = []

        ensure_dir(RESULTS_DIR)

        for person_name in os.listdir(PROBE_DIR):
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

                    retrieved = [m for m in metadatas[0] if m is not None]
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

        json_path = os.path.join(RESULTS_DIR, f"evaluation_results_{INDEX_TYPE}_{METRIC}_k{K}.json")
        csv_path = os.path.join(RESULTS_DIR, f"evaluation_summary_{INDEX_TYPE}_{METRIC}_k{K}.csv")
        ensure_dir(RESULTS_DIR)

        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        pd.DataFrame(results).to_csv(csv_path, index=False)

        total = len(results)
        correct = sum(r["top_k_correct"] for r in results)
        accuracy = correct / total if total > 0 else 0

        print(f"\n=== Summary for {INDEX_TYPE.upper()} | Metric: {METRIC} | K: {K} ===")
        print(f"Total probes: {total}")
        print(f"Top-{K} accuracy: {accuracy*100:.2f}%")

        print(f"Results saved to {json_path} and {csv_path}")

    else:
        # === New evaluation mode: multiple m values ===
        summary_data = []

        for m in m_values:
            print(f"\n=== Evaluating for m = {m} ===")

            index = index_class(dim=index_dim, metric=METRIC if INDEX_TYPE != "lsh" else None)
            index = build_gallery_subset(preprocessor, model, index, m)
            searcher = FaissSearch(index, metric=METRIC)

            results = []

            for person_name in os.listdir(PROBE_DIR):
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

                        retrieved = [m for m in metadatas[0] if m is not None]
                        top_k_correct = true_identity in retrieved[:K]

                        results.append({
                            "probe_image": probe_file,
                            "true_identity": true_identity,
                            "retrieved": retrieved,
                            "distances": distances[0].tolist(),
                            "time_sec": elapsed,
                            "top_k_correct": top_k_correct,
                            "m_value": m
                        })

                        print(f"Probe {probe_file}: top-{K} correct? {top_k_correct} (retrieved: {retrieved})")

                    except Exception as e:
                        print(f"Error processing probe {probe_file}: {e}")

            total = len(results)
            correct = sum(r["top_k_correct"] for r in results)
            accuracy = correct / total if total > 0 else 0

            print(f"\n>>> m={m} | Top-{K} accuracy: {accuracy*100:.2f}% <<<\n")

            result_dir = os.path.join("evaluation_results_m", MODEL_NAME, INDEX_TYPE, METRIC, f"m{m}", f"k{K}")
            json_path = os.path.join(result_dir, f"evaluation_results_{INDEX_TYPE}_{METRIC}_m{m}_k{K}.json")
            csv_path = os.path.join(result_dir, f"evaluation_summary_{INDEX_TYPE}_{METRIC}_m{m}_k{K}.csv")

            # Ensure result directory exists before saving files
            ensure_dir(result_dir)

            with open(json_path, "w") as f:
                json.dump(results, f, indent=4)
            pd.DataFrame(results).to_csv(csv_path, index=False)
            
            summary_data.append({
                "m": m,
                "total_probes": total,
                "correct": correct,
                "top_k_accuracy": accuracy
            })

        summary_csv_dir = os.path.join("evaluation_results_m", MODEL_NAME, INDEX_TYPE, METRIC)
        summary_csv = os.path.join(summary_csv_dir, f"summary_k{K}.csv")
        ensure_dir(summary_csv_dir)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)

        print(f"\nAll m evaluations complete. Summary saved to {summary_csv}")


if __name__ == "__main__":
    # To run normal evaluation on full gallery, just call:
    # evaluate()

    # To run evaluation for multiple m values (subset gallery size), call:
    evaluate(m_values=list(range(1, 6)))