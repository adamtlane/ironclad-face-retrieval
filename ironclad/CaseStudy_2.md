# Second Case Study for the IronClad System
## Adam Lane

*IronClad* is a modular system for extracting, embedding, and indexing visual data. It was designed to support large-scale face and headshot comparisons, all while leveraging various FAISS index methods to store vector embeddings alongside the associated metadata. By doing so, the system can enable efficient similarity search across high-dimensional image features. With support for multiple search backends and FAISS indexing methods (brute-force, HNSW, and LSH), *IronClad* provides a scalable and extensible foundation for visual recognition and retrieval tasks.

## **Task 1: Embedding Model Comparison**

This task evaluates the performance of two popular distance metrics (Cosine similarity and Euclidean distance) using `bruteforce` nearest neighbor search on two well-known facial recognition datasets, `casia-webface` and `vggface2`. The goal is to measure and compare their effectiveness using four retrieval metrics - Top-K Accuracy, Mean Reciprocal Rank (MRR), mean Average Precision (mAP), and Recall (per the specific K value). Experiments were conducted at K = 1, 3, and 5, and results were aggregated across datasets and metrics. The analysis shows that cosine and Euclidean distance perform identically across all tested configurations. For each dataset and Top-K value, they yield the exact same results on all four metrics, including mAP and MRR. This suggests that for the given embedding models and datasets, both distance metrics are equally suitable in `bruteforce` retrieval contexts. These results can be seen tabulated below:

<div align="center">

| Dataset       | Index Type | Distance  | Top-K | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| ------------- | ---------- | --------- | ----- | ------------------ | ------ | ------ | ------------- |
| vggface2      | bruteforce | cosine    | 1     | 50.65              | 0.5065 | 0.5065 | 50.65         |
| vggface2      | bruteforce | euclidean | 1     | 50.65              | 0.5065 | 0.5065 | 50.65         |
| casia-webface | bruteforce | cosine    | 1     | 9.31               | 0.0931 | 0.0931 | 9.31          |
| casia-webface | bruteforce | euclidean | 1     | 9.31               | 0.0931 | 0.0931 | 9.31          |

</div>

<p align="center"><b>Table 1: Embedding Model Comparison Results (Top-1).</b></p>

<div align="center">

| Dataset       | Index Type | Distance  | Top-K | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| ------------- | ---------- | --------- | ----- | ------------------ | ------ | ------ | ------------- |
| vggface2      | bruteforce | cosine    | 3     | 18.12              | 0.1380 | 0.1369 | 18.12         |
| vggface2      | bruteforce | euclidean | 3     | 18.12              | 0.1380 | 0.1369 | 18.12         |
| casia-webface | bruteforce | cosine    | 3     | 6.81               | 0.0539 | 0.0537 | 6.81          |
| casia-webface | bruteforce | euclidean | 3     | 6.81               | 0.0539 | 0.0537 | 6.81          |

</div>

<p align="center"><b>Table 2: Embedding Model Comparison Results (Top-3).</b></p>

<div align="center">

| Dataset       | Index Type | Distance  | Top-K | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| ------------- | ---------- | --------- | ----- | ------------------ | ------ | ------ | ------------- |
| vggface2      | bruteforce | cosine    | 5     | 66.47              | 0.5658 | 0.5551 | 66.47         |
| vggface2      | bruteforce | euclidean | 5     | 66.47              | 0.5658 | 0.5551 | 66.47         |
| casia-webface | bruteforce | cosine    | 5     | 16.32              | 0.1177 | 0.1161 | 16.32         |
| casia-webface | bruteforce | euclidean | 5     | 16.32              | 0.1177 | 0.1161 | 16.32         |

</div>

<p align="center"><b>Table 3: Embedding Model Comparison Results (Top-5).</b></p>

In terms of dataset performance, `vggface2` significantly outperforms `casia-webface`. For example, at `K=1`, `vggface2` achieves 50.65% accuracy compared to just 9.31% for `casia-webface`. This trend persists at higher values of K, with `vggface2` consistently achieving over 66% accuracy at `K=5`. These differences likely stem from dataset quality, variation, and the strength of the associated embedding models rather than the distance metric used. Although mAP and MRR were included to provide deeper ranking insight, their values closely track Top-K Accuracy and Recall, reinforcing the same conclusions. For instance, when accuracy improves at higher K, both mAP and MRR increase proportionally, indicating consistency between ranking-based and hit-based evaluation metrics.

Since both cosine similarity and Euclidean distance yield equivalent retrieval performance in `bruteforce` settings, the choice between them should be guided by implementation preferences. Cosine similarity is generally favored in face recognition tasks because it compares the angle between vectors, which is more robust to scale differences, especially relevant when using L2-normalized embeddings (which is common in facial models). Therefore, despite equal performance here, cosine similarity is recommended for Tasks 2–5 due to its robustness, widespread use in embedding-based retrieval, and better interpretability in high-dimensional similarity comparisons. While cosine and Euclidean perform the same in brute-force search, it is worth verifying whether this holds under approximate indexing methods such as `hnsw` or `lsh`, where internal mechanics might favor one metric over the other. Additionally, the large performance gap between datasets underscores the importance of choosing high-quality datasets and embeddings when building face retrieval systems. To better view these results visually, the three figures below reveal this trend as well:

<p align="center">
  <img src="external_plots_m/Task%201%20-%20Plot%201.png" width="750"/>
</p>

<p align="center"><b>Figure 1: Retrieval Metrics on Brute-Force (Top-1).</b></p>

<p align="center">
  <img src="external_plots_m/Task%201%20-%20Plot%202.png" width="750"/>
</p>

<p align="center"><b>Figure 2: Retrieval Metrics on Brute-Force (Top-3).</b></p>

<p align="center">
  <img src="external_plots_m/Task%201%20-%20Plot%203.png" width="750"/>
</p>

<p align="center"><b>Figure 3: Retrieval Metrics on Brute-Force (Top-5).</b></p>

## **Task 2: Impact of Gallery Size on CASIA-WebFace Retrieval**

To investigate the effect of gallery size on retrieval performance, we evaluated `casia-webface` across different values of `m`, the number of images per identity included in the gallery. The value of `m` was varied from 1 to 5, with retrieval performance measured using Top-3 Accuracy, Mean Reciprocal Rank (MRR), Mean Average Precision (mAP), and Recall\@3. For both cosine and Euclidean distance metrics using `bruteforce` search, performance generally improved as `m` increased. In the cosine case, Top-3 Accuracy rose steadily from 11.61% at `m=1` to a peak of 13.51% at `m=4`. Both MRR and mAP followed a similar trend, indicating more accurate and confident retrievals. However, further increasing `m` to 5 yielded negligible or slightly reduced gains, suggesting a plateau in performance. Euclidean distance exhibited similar behavior, with peak metrics occurring between `m=3` and `m=5`, but with slightly lower performance overall compared to cosine. This can be seen in the two tables below:

<div align="center">

| Dataset       | Index Type | Distance  | Top-K | m | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| ------------- | ---------- | --------- | ----- | - | ------------------ | ------ | ------ | ------------- |
| casia-webface | bruteforce | cosine    | 3     | 1 | 11.61              | 0.0943 | 0.0943 | 11.61         |
| casia-webface | bruteforce | cosine    | 3     | 2 | 12.71              | 0.1006 | 0.1004 | 12.71         |
| casia-webface | bruteforce | cosine    | 3     | 3 | 13.11              | 0.1068 | 0.1066 | 13.11         |
| casia-webface | bruteforce | cosine    | 3     | 4 | 13.51              | 0.1093 | 0.1084 | 13.51         |
| casia-webface | bruteforce | cosine    | 3     | 5 | 13.41              | 0.1079 | 0.1080 | 13.41         |

</div>

<p align="center"><b>Table 4: CASIA-WebFace Retrieval Results (Cosine).</b></p>

<div align="center">

| Dataset       | Index Type | Distance  | Top-K | m | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| ------------- | ---------- | --------- | ----- | - | ------------------ | ------ | ------ | ------------- |
| casia-webface | bruteforce | euclidean | 3     | 1 | 11.11              | 0.0909 | 0.0909 | 11.11         |
| casia-webface | bruteforce | euclidean | 3     | 2 | 12.81              | 0.1038 | 0.1036 | 12.81         |
| casia-webface | bruteforce | euclidean | 3     | 3 | 13.01              | 0.1064 | 0.1061 | 13.01         |
| casia-webface | bruteforce | euclidean | 3     | 4 | 12.71              | 0.1048 | 0.1044 | 12.71         |
| casia-webface | bruteforce | euclidean | 3     | 5 | 12.81              | 0.1061 | 0.1060 | 12.81         |

</div>

<p align="center"><b>Table 5: CASIA-WebFace Retrieval Results (Euclidean).</b></p>


These results suggest that increasing the number of gallery images improves the model's ability to handle intra-class variation in facial appearance, such as changes in pose, lighting, and expression. However, the benefits taper off around `m=4`, likely due to diminishing returns from additional (and possibly redundant) images. We therefore recommend using m = 4 as an optimal value for `casia-webface`, as it balances accuracy gains with computational cost. Dataset-specific factors, such as variation in image quality and the natural diversity of `casia-webface` subjects, also play a role. For instance, beyond four representative images, new entries may add little value and can even introduce noise. Moreover, the distribution of available images per identity may limit the feasibility of consistently increasing `m` across the dataset. In summary, our experiments show that moderate increases in `m` can significantly boost retrieval performance, but the effect saturates quickly. Using four images per identity provides a practical and effective trade-off:

<p align="center">
  <img src="external_plots_m/Task%202%20-%20Plot%201.png" width="750"/>
</p>

<p align="center"><b>Figure 4: Retrieval Metrics on CASIA-WebFace (Cosine).</b></p>

<p align="center">
  <img src="external_plots_m/Task%202%20-%20Plot%202.png" width="750"/>
</p>

<p align="center"><b>Figure 5: Retrieval Metrics on CASIA-WebFace (Euclidean).</b></p>

## **Task 3: Impact of Gallery Size on VGGFace2 Retrieval**

As the number of gallery images per identity (`m`) increases from 1 to 5, retrieval performance on `vggface2` improves across all metrics, with the most significant gains occurring between `m=1` and `m=3`. Using cosine distance, Top-3 accuracy rises from 58.26% (m=1) to 62.86% (m=3), while euclidean distance yields a more gradual improvement, reaching 63.46% at `m=5`. These gains reflect `vggface2`'s high intra-class variation; more gallery images better capture this variability, increasing the chance of a correct match. However, returns diminish beyond `m=3` for cosine distance, suggesting a balance point between performance and efficiency. Given these findings, `m=3` is recommended for most systems as it provides strong performance without excessive resource demands. If maximum accuracy is critical and resources allow, `m=5` is preferable. Finally, `vggface2` embeddings outperform `casia-webface` across all settings, confirming it as the more robust model for deployment in face retrieval applications. Again, the tabulated results can be seen below:

<div align="center">

| Dataset  | Index Type | Distance  | Top-K | m | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| -------- | ---------- | --------- | ----- | - | ------------------ | ------ | ------ | ------------- |
| vggface2 | bruteforce | cosine    | 3     | 1 | 58.26              | 0.5070 | 0.5070 | 58.26         |
| vggface2 | bruteforce | cosine    | 3     | 2 | 62.56              | 0.5487 | 0.5445 | 62.56         |
| vggface2 | bruteforce | cosine    | 3     | 3 | 62.86              | 0.5536 | 0.5501 | 62.86         |
| vggface2 | bruteforce | cosine    | 3     | 4 | 62.46              | 0.5611 | 0.5566 | 62.46         |
| vggface2 | bruteforce | cosine    | 3     | 5 | 62.76              | 0.5581 | 0.5534 | 62.76         |

</div>

<p align="center"><b>Table 6: VGGFace2 Retrieval Results (Cosine).</b></p>

<div align="center">

| Dataset  | Index Type | Distance  | Top-K | m | Top-K Accuracy (%) | MRR    | mAP    | Recall\@K (%) |
| -------- | ---------- | --------- | ----- | - | ------------------ | ------ | ------ | ------------- |
| vggface2 | bruteforce | euclidean | 3     | 1 | 58.16              | 0.5052 | 0.5052 | 58.16         |
| vggface2 | bruteforce | euclidean | 3     | 2 | 61.86              | 0.5434 | 0.5396 | 61.86         |
| vggface2 | bruteforce | euclidean | 3     | 3 | 62.26              | 0.5531 | 0.5496 | 62.26         |
| vggface2 | bruteforce | euclidean | 3     | 4 | 62.66              | 0.5571 | 0.5525 | 62.66         |
| vggface2 | bruteforce | euclidean | 3     | 5 | 63.46              | 0.5621 | 0.5574 | 63.46         |

</div>

<p align="center"><b>Table 7: VGGFace2 Retrieval Results (Euclidean).</b></p>

Across all configurations, `vggface2` vastly outperforms `casia-webface` in face retrieval tasks. For example, at `m=1`, `vggface2` achieves a Top-3 accuracy of \~58%, while `casia-webface` struggles at \~11%. This performance gap persists consistently across all `m` values and both distance metrics. The `vggface2` model demonstrates not only higher base accuracy but also a more substantial and consistent benefit from increasing `m`. While `casia-webface`'s performance saturates quickly (with only marginal gains beyond `m=3`), VGGFace2 continues to improve steadily up to `m=5`, particularly with Euclidean distance. It should be noted that `vggface2` was trained on a much larger and more diverse dataset (millions of images with wide pose, age, and lighting variations), enabling better generalization and robustness to real-world conditions. By comparison, `casia-webface` is comparatively narrower in scope and variety.

The `vggface2` embeddings were produced using deeper, more expressive models (such as ResNet-50) compared to the simpler CNNs used with `casia-webface`. This yields more discriminative features for retrieval. The cosine and Euclidean distances in `vggface2` space consistently align with semantic identity, reflecting well-structured embedding spaces. `casia-webface` embeddings are less separable, causing more frequent retrieval confusion. With `casia-webface`, increasing `m` shows diminishing returns—suggesting that the embeddings themselves are a limiting factor. For `vggface2`, more gallery images directly enhance robustness, especially for individuals with high appearance variability. If system performance is a priority, `vggface2` is the clearly superior embedding model. It consistently outperforms `casia-webface` across all metrics and gallery sizes, benefits more meaningfully from increased samples per identity, and better handles the variability present in probe images. For practical deployments, `vggface2` with `m=3–5` strikes a strong balance between accuracy and efficiency:

<p align="center">
  <img src="external_plots_m/Task%203%20-%20Plot%201.png" width="750"/>
</p>

<p align="center"><b>Figure 6: Retrieval Metrics on VGGFace2 (Cosine).</b></p>

<p align="center">
  <img src="external_plots_m/Task%203%20-%20Plot%202.png" width="750"/>
</p>

<p align="center"><b>Figure 7: Retrieval Metrics on VGGFace2 (Euclidean).</b></p>

## **Task 4: Brute-Force vs HNSW Indexing**

We evaluated the impact of index type (`bruteforce` and `hnsw`) on retrieval performance using two embedding models (`casia-webface` and `vggface2`), with cosine distance, `k=3`, and `m=3` fixed. Across both datasets, `bruteforce` achieved slightly higher accuracy, though the margin was minimal. For example, `vggface2`'s Top-3 accuracy was 62.86% using `bruteforce` and 62.06% with `hnsw`. On `casia-webface`, the difference was similarly small, only 13.41% vs 13.11%. In terms of retrieval quality, both methods produced nearly identical MRR and mAP scores, indicating that `hnsw`'s approximate search retains much of the precision of `bruteforce`. This consistency across all evaluation metrics shows that `hnsw` is a viable alternative for high-accuracy retrieval while being more efficient:

<div align="center">

| Dataset       | Index Type | Top-K Accuracy (%) | MRR    | mAP    | Recall\@3 (%) | Avg Query Time (s) | Min Query Time (s) | Max Query Time (s) |
| ------------- | ---------- | ------------------ | ------ | ------ | ------------- | ------------------ | ------------------ | ------------------ |
| vggface2      | bruteforce | 62.86              | 0.5536 | 0.5501 | 62.86         | 0.0493             | 0.0390             | 0.1339             |
| vggface2      | hnsw       | 62.06              | 0.5534 | 0.5486 | 62.06         | 0.0504             | 0.0435             | 0.0838             |
| casia-webface | bruteforce | 13.41              | 0.1054 | 0.1054 | 13.41         | 0.0584             | 0.0383             | 0.1597             |
| casia-webface | hnsw       | 13.11              | 0.1051 | 0.1049 | 13.11         | 0.0583             | 0.0430             | 0.1135             |

</div>

<p align="center"><b>Table 8: Brute-Force and HNSW Comparison Results.</b></p>


Average query times were also similar at current scale, hovering around 0.05–0.06 seconds per probe for both index types. However, `bruteforce` search time increases linearly with dataset size, while HNSW scales sublinearly due to its graph-based structure. When projected to a billion-image gallery, Brute Force could require minutes per query, whereas `hnsw` is expected to handle such scale in milliseconds to seconds, depending on tuning. Assuming `vggface2`'s `bruteforce` average of 0.0493 s/query on ~12K gallery entries: A billion-image gallery would be ~80,000× larger. `bruteforce` could scale to ~66 minutes/query, which is unfeasible for real-world use. By contrast, `hnsw`'s sublinear scaling suggests it can handle billion-scale searches in milliseconds to seconds, depending on graph construction and tuning (Such as `ef` and `M`). Given the negligible accuracy trade-off and massive gains in efficiency at scale, we conclude that `hnsw` is the preferred indexing method for real-world deployment. It enables scalable, fast retrieval with accuracy that closely matches `bruteforce`, even when working with much larger datasets:

<p align="center">
  <img src="external_plots_m/Task%204%20-%20Plot%201.png" width="750"/>
</p>

<p align="center"><b>Figure 8: Brute Force and HNSW Retrieval Comparison.</b></p>

## **Task 5: HNSW vs LSH Indexing**

Lastly, we evaluated the impact of selecting `hnsw` versus `lsh` as the indexing method for face retrieval across both `vggface2` and `casia-webface` datasets, holding all other parameters constant (`m=3`, Top-`K=3`, cosine distance). The results show a notable trade-off between retrieval accuracy and indexing strategy. On `vggface2`, `hnsw` achieved a Top-3 accuracy of 62.06%, with an MRR of 0.5534 and mAP of 0.5486, while `lsh` lagged significantly behind at 46.45% Top-3 accuracy, 0.3907 MRR, and 0.3891 mAP. A similar trend was observed on `casia-webface`, where `hnsw` outperformed `lsh` at 13.11% vs 6.71% Top-3 accuracy. These results affirm that `hnsw` maintains strong retrieval performance, while `lsh` sacrifices accuracy considerably for approximate similarity:

<div align="center">

| Dataset       | Index Type | Top-K Accuracy (%) | MRR    | mAP    | Recall\@3 (%) | Avg Query Time (s) | Min Query Time (s) | Max Query Time (s) |
| ------------- | ---------- | ------------------ | ------ | ------ | ------------- | ------------------ | ------------------ | ------------------ |
| vggface2      | hnsw       | 62.06              | 0.5534 | 0.5486 | 62.06         | 0.0504             | 0.0435             | 0.0838             |
| vggface2      | lsh        | 46.45              | 0.3907 | 0.3891 | 46.45         | 0.0512             | 0.0377             | 0.0919             |
| casia-webface | hnsw       | 13.11              | 0.1051 | 0.1049 | 13.11         | 0.0583             | 0.0430             | 0.1135             |
| casia-webface | lsh        | 6.71               | 0.0462 | 0.0463 | 6.71          | 0.0499             | 0.0395             | 0.1103             |

</div>

<p align="center"><b>Table 9:HNSW and LSH Comparison Results.</b></p>

Interestingly, the query time differences between `hnsw` and `lsh` were negligible, with average times in the \~0.05 second range across both datasets and methods. This suggests that the computational efficiency advantage of `lsh` is not clearly realized at this scale or implementation, whereas its accuracy degradation is more pronounced. For large-scale systems (such as billions of images), `lsh` may still be considered when extremely fast, memory efficient indexing is prioritized over accuracy. This is especially true in scenarios where approximate results are acceptable, such as real time pre-filtering or resource constrained environments. However, for use cases that demand higher retrieval precision and robustness, `hnsw` remains the preferred option, offering a better balance between speed and accuracy:

<p align="center">
  <img src="external_plots_m/Task%205%20-%20Plot%201.png" width="750"/>
</p>

<p align="center"><b>Figure 9: HNSW and LSH Retrieval Comparison.</b></p>