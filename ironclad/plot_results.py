import matplotlib.pyplot as plt
import numpy as np

# Helper function to add labels above bars
def autolabel(rects, ax, decimals=2):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.{decimals}f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

# Define the plotting function
def plot_grouped_metrics(data, top_k):
    labels = [f"{row['Dataset']}-{row['Distance']}" for row in data]
    topk_acc = [row["Top-K Accuracy (%)"] for row in data]
    mrr = [row["MRR"] for row in data]
    mAP = [row["mAP"] for row in data]
    recall = [row["Recall@K (%)"] for row in data]

    x = np.arange(len(labels))  # label locations
    width = 0.2  # width of each bar

    fig, ax = plt.subplots(figsize=(16, 9))
    rects1 = ax.bar(x - 1.5*width, topk_acc, width, label='Top-K Accuracy')
    rects2 = ax.bar(x - 0.5*width, mrr, width, label='MRR')
    rects3 = ax.bar(x + 0.5*width, mAP, width, label='mAP')
    rects4 = ax.bar(x + 1.5*width, recall, width, label='Recall@K')

    # Log scale for y-axis
    ax.set_ylabel('Objective Metric Score')
    ax.set_title(f"Retrieval Metrics on Brute Force (Top-{top_k})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # Add metric values above bars
    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    autolabel(rects4, ax)

    plt.tight_layout()
    plt.show()

# === DATA SETUP ===

data_k1 = [
    {"Dataset": "vggface2", "Distance": "cosine", "Top-K Accuracy (%)": 50.65, "MRR": 0.5065, "mAP": 0.5065, "Recall@K (%)": 50.65},
    {"Dataset": "vggface2", "Distance": "euclidean", "Top-K Accuracy (%)": 50.65, "MRR": 0.5065, "mAP": 0.5065, "Recall@K (%)": 50.65},
    {"Dataset": "casia-webface", "Distance": "cosine", "Top-K Accuracy (%)": 9.31, "MRR": 0.0931, "mAP": 0.0931, "Recall@K (%)": 9.31},
    {"Dataset": "casia-webface", "Distance": "euclidean", "Top-K Accuracy (%)": 9.31, "MRR": 0.0931, "mAP": 0.0931, "Recall@K (%)": 9.31},
]

data_k3 = [
    {"Dataset": "vggface2", "Distance": "cosine", "Top-K Accuracy (%)": 18.12, "MRR": 0.1380, "mAP": 0.1369, "Recall@K (%)": 18.12},
    {"Dataset": "vggface2", "Distance": "euclidean", "Top-K Accuracy (%)": 18.12, "MRR": 0.1380, "mAP": 0.1369, "Recall@K (%)": 18.12},
    {"Dataset": "casia-webface", "Distance": "cosine", "Top-K Accuracy (%)": 6.81, "MRR": 0.0539, "mAP": 0.0537, "Recall@K (%)": 6.81},
    {"Dataset": "casia-webface", "Distance": "euclidean", "Top-K Accuracy (%)": 6.81, "MRR": 0.0539, "mAP": 0.0537, "Recall@K (%)": 6.81},
]

data_k5 = [
    {"Dataset": "vggface2", "Distance": "cosine", "Top-K Accuracy (%)": 66.47, "MRR": 0.5658, "mAP": 0.5551, "Recall@K (%)": 66.47},
    {"Dataset": "vggface2", "Distance": "euclidean", "Top-K Accuracy (%)": 66.47, "MRR": 0.5658, "mAP": 0.5551, "Recall@K (%)": 66.47},
    {"Dataset": "casia-webface", "Distance": "cosine", "Top-K Accuracy (%)": 16.32, "MRR": 0.1177, "mAP": 0.1161, "Recall@K (%)": 16.32},
    {"Dataset": "casia-webface", "Distance": "euclidean", "Top-K Accuracy (%)": 16.32, "MRR": 0.1177, "mAP": 0.1161, "Recall@K (%)": 16.32},
]

# === PLOT ALL ===
plot_grouped_metrics(data_k1, top_k=1)
plot_grouped_metrics(data_k3, top_k=3)
plot_grouped_metrics(data_k5, top_k=5)