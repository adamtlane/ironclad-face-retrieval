import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INDEX_TYPE = "lsh" # Configuration option: "lsh", "hnsw", or "bruteforce". Determines index type.
METRIC = "cosine" # Configuration option: "cosine" or "euclidean". Determines distance metric.
TOP_K = 3 # Configuration option: 1, 3, or 5. Top-K results to retrieve.
MODEL_NAME = "vggface2" # Configuration option: `vggface2`or `casia-webface`. Pretrained embedding model name.

M_VALUE = 3
RESULTS_DIR = os.path.join("evaluation_results_m", MODEL_NAME, INDEX_TYPE, METRIC, f"m{M_VALUE}", f"k{TOP_K}")

JSON_RESULTS_PATH = os.path.join(RESULTS_DIR, f"evaluation_results_{INDEX_TYPE}_{METRIC}_m{M_VALUE}_k{TOP_K}.json")
CSV_SUMMARY_PATH = os.path.join(RESULTS_DIR, f"evaluation_summary_{INDEX_TYPE}_{METRIC}_m{M_VALUE}_k{TOP_K}.csv")

PLOTS_DIR = os.path.join("plots_m", MODEL_NAME, INDEX_TYPE, METRIC, f"m{M_VALUE}", f"k{TOP_K}") # Path for saving generated plots.

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    """
    Load evaluation results from CSV (preferred) or fallback to JSON.

    Returns:
        pd.DataFrame: DataFrame containing results with a 'top_k_correct' boolean column.
    """
    if os.path.exists(CSV_SUMMARY_PATH): # Check to see if the CSV file path exists.
        df = pd.read_csv(CSV_SUMMARY_PATH)
        print(f"Loaded {len(df)} entries from {CSV_SUMMARY_PATH}")
    elif os.path.exists(JSON_RESULTS_PATH):
        with open(JSON_RESULTS_PATH, "r") as f: # Perform a similar check for the JSON files.
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} entries from {JSON_RESULTS_PATH}") 
    else:
        raise FileNotFoundError("Neither CSV nor JSON results were found.")

    if 'top_k_correct' in df.columns: # Ensure the column 'top_k_correct' is interpreted as a boolean.
        df['top_k_correct'] = df['top_k_correct'].astype(bool)
    else:
        raise KeyError("'top_k_correct' column is missing from results.")

    return df

def plot_top_k_accuracy(df):
    """
    Generate and save a bar plot showing overall Top-K accuracy.

    Args:
        df (pd.DataFrame): DataFrame with evaluation results.
    """
    accuracy = df["top_k_correct"].mean() # Calculate the accuracy.

    print(f"Top-{TOP_K} accuracy: {accuracy*100:.2f}% ({df['top_k_correct'].sum()} / {len(df)})")

    plt.figure(figsize=(6, 4))
    sns.barplot(x=[f"Top-{TOP_K} Accuracy"], y=[accuracy])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Overall Top-{TOP_K} Accuracy")
    plt.savefig(os.path.join(PLOTS_DIR, "top_k_accuracy.png"))
    plt.close()

def plot_search_time_distribution(df):
    """
    Plot and save histogram showing distribution of search times.

    Args:
        df (pd.DataFrame): DataFrame with evaluation results.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(df["time_sec"], bins=30, kde=True)
    plt.title("Search Time Distribution (seconds)")
    plt.xlabel("Time (seconds)")
    plt.savefig(os.path.join(PLOTS_DIR, "search_time_distribution.png"))
    plt.close()

def plot_distance_boxplots(df):
    """
    Create a boxplot comparing first retrieval distances for correct vs. incorrect matches.

    Args:
        df (pd.DataFrame): DataFrame with evaluation results.
    """
    
    df['first_distance'] = df['distances'].apply(
        lambda x: float(x.strip('[]').split(',')[0]) if isinstance(x, str) else x[0]
    ) # Extract the first distance value from each list.

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="top_k_correct", y="first_distance", data=df)
    plt.title("Distance Distribution: Correct vs Incorrect Retrievals")
    plt.xlabel("Top-K Correct")
    plt.ylabel("Distance (Cosine)")
    plt.savefig(os.path.join(PLOTS_DIR, "distance_correct_vs_incorrect.png"))
    plt.close()

def plot_identity_counts(df):
    """
    Plot and save a bar chart showing most frequently retrieved identities.

    Args:
        df (pd.DataFrame): DataFrame with evaluation results.
    """
    from collections import Counter

    all_retrieved = []
    
    for retrieved_list in df['retrieved']: # Flatten all retrieved identities into a single list.
        if isinstance(retrieved_list, str):
            identities = retrieved_list.strip("[]").replace("'", "").split(", ")
            all_retrieved.extend(identities)
        else:
            all_retrieved.extend(retrieved_list)

    counter = Counter(all_retrieved) # Count most common retrieved identities.
    most_common = counter.most_common(20)
    names, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(names), y=list(counts))
    plt.xticks(rotation=45, ha='right')
    plt.title("Top 20 Most Frequently Retrieved Identities in Top-K")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_retrieved_identities.png"))
    plt.close()

def compute_mrr(df):
    reciprocal_ranks = []

    for _, row in df.iterrows():
        retrieved = eval(row['retrieved']) if isinstance(row['retrieved'], str) else row['retrieved']
        true_id = row['true_identity']
        if true_id in retrieved:
            rank = retrieved.index(true_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

def compute_map(df):
    def average_precision(retrieved, true_label):
        hits = 0
        sum_precisions = 0
        for i, label in enumerate(retrieved):
            if label == true_label:
                hits += 1
                sum_precisions += hits / (i + 1)
        return sum_precisions / hits if hits > 0 else 0

    average_precisions = []
    for _, row in df.iterrows():
        retrieved = eval(row['retrieved']) if isinstance(row['retrieved'], str) else row['retrieved']
        true_id = row['true_identity']
        average_precisions.append(average_precision(retrieved, true_id))

    return sum(average_precisions) / len(average_precisions)

def compute_recall_at_k(df):
    """
    Compute Recall@K for retrieval.

    Args:
        df (pd.DataFrame): DataFrame containing 'top_k_correct' boolean column.

    Returns:
        float: Recall@K value.
    """
    return df["top_k_correct"].mean() if "top_k_correct" in df.columns else 0.0

def main():
    """
    Main execution routine to load data, compute Top-K accuracy, generate plots,
    and save summary text file with results.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_data()
    summary_path = os.path.join(PLOTS_DIR, "summary.txt")
    
    with open(summary_path, "w") as f: # Create or overwrite summary log file.
        def log(msg):
            print(msg)
            f.write(msg + "\n")

        correct = df["top_k_correct"].sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0

        log(f"Top-{TOP_K} Accuracy: {accuracy*100:.2f}% ({correct}/{total})")
        
        plot_top_k_accuracy(df) # Generate all related plots.
        plot_search_time_distribution(df)
        plot_distance_boxplots(df)
        plot_identity_counts(df)

        log(f"Plots and summary saved to: {PLOTS_DIR}")

        mrr = compute_mrr(df)
        map_score = compute_map(df)

        log(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        log(f"Mean Average Precision (mAP): {map_score:.4f}")

        recall_at_k = compute_recall_at_k(df)
        log(f"Recall@{TOP_K}: {recall_at_k:.4f} ({recall_at_k*100:.2f}%)")

        average_time = df["time_sec"].mean()
        max_time = df["time_sec"].max()
        min_time = df["time_sec"].min()

        print(f"Avg Query Time: {average_time:.4f} sec")
        print(f"Min Query Time: {min_time:.4f} sec")
        print(f"Max Query Time: {max_time:.4f} sec")

if __name__ == "__main__":
    main()