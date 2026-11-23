# Case Study for the IronClad System
## Adam Lane

*IronClad* is a modular system for extracting, embedding, and indexing visual data. It was designed to support large-scale face and headshot comparisons, all while leveraging various FAISS index methods to store vector embeddings alongside the associated metadata. By doing so, the system can enable efficient similarity search across high-dimensional image features. With support for multiple search backends and FAISS indexing methods (brute-force, HNSW, and LSH), *IronClad* provides a scalable and extensible foundation for visual recognition and retrieval tasks.

## **Task 1: Standard Embedding Model Comparison**

To evaluate the performance of the two available embedding models, `casia-webface` and `vggface2`, we conducted Top-3 nearest neighbor searches using the three indexing strategies (`bruteforce`, `hnsw`, and `lsh`) and two distance metrics (cosine similarity and Euclidean distance). The results are summarized in Table 1, shown below:

<div align="center">

| Model         | Search Method | Metric    | Top-3 Accuracy | Correct Predictions |
| ------------- | ------------- | --------- | -------------- | ------------ |
| casia-webface | bruteforce    | Cosine    | 13.61%         | 136 / 999    |
| casia-webface | bruteforce    | Euclidean | 13.61%         | 136 / 999    |
| casia-webface | hnsw          | Cosine    | 13.51%         | 135 / 999    |
| casia-webface | hnsw          | Euclidean | 13.51%         | 135 / 999    |
| casia-webface | lsh           | Cosine    | 6.51%          | 65 / 999     |
| casia-webface | lsh           | Euclidean | 6.51%          | 65 / 999     |
| vggface2      | bruteforce    | Cosine    | 61.96%         | 619 / 999    |
| vggface2      | bruteforce    | Euclidean | 61.96%         | 619 / 999    |
| vggface2      | hnsw          | Cosine    | 61.76%         | 617 / 999    |
| vggface2      | hnsw          | Euclidean | 61.76%         | 617 / 999    |
| vggface2      | lsh           | Cosine    | 46.35%         | 463 / 999    |
| vggface2      | lsh           | Euclidean | 46.35%         | 463 / 999    |

</div>

<p align="center"><b>Table 1: Top-3 Accuracy (%) for Each Embedding Model by Search Method and Distance Metric.</b></p>

When averaging Top-3 accuracy across all configurations, `vggface2` achieves approximately 57.0%, whereas `casia-webface` reaches only 11.9%. Thus, `vggface2` consistently outperforms Casia-Webface by a substantial margin, often exceeding four times the accuracy. This indicates that `vggface2` produces more discriminative and reliable face embeddings for the dataset provided. The specific impact that the search method has on the results can be seen in the accompanying table:

<div align="center">

| Model         | bruteforce | hnsw   | lsh    |
| ------------- | ---------- | ------ | ------ |
| casia-webface | 13.61%     | 13.51% | 6.51%  |
| vggface2      | 61.96%     | 61.76% | 46.35% |

</div>

<p align="center"><b>Table 2: Impact of Search Method.</b></p>

For both models, `bruteforce` and `hnsw` perform nearly identically, indicating that `hnsw` provides efficient indexing without significant accuracy loss. However, performance degrades for both models under `lsh`, particularly for `casia-webface`, which suffers a drop of over 50% in Top-3 accuracy. This suggests that Casia’s embeddings are more vulnerable to approximation noise and less separable in the LSH-projected space. It should be noted that there is no observable performance difference between cosine similarity and Euclidean distance across any configuration. This suggests the embeddings from both models are either normalized or structured such that both metrics yield equivalent neighborhood structures. This is also best shown visually, as can be observed in Figure 1, displaying the Top-3 accuracy values grouped by distance method:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%201%2C%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 1: Top-3 Accuracy Results by Distance.</b></p>

Our `vggface2` model shows strong robustness to changes in indexing methods and distance metrics. Even under LSH—a more aggressive approximation strategy—it maintains relatively high performance (Top-3 accuracy above 46%). By contrast, `casia-webface` degrades significantly under the same conditions, suggesting it is less suitable for efficient retrieval tasks in real-world scenarios. Overall, `vggface2` is the clearly superior embedding model for the current task. It delivers high Top-3 accuracy, exhibits minimal sensitivity to indexing method or metric, and scales well across different retrieval strategies. However, `casia-webface` underperforms and demonstrates fragility in approximate search conditions. For any production deployment or further experimentation, `vggface2` is the recommended foundation. The choice becomes even more apparent when comparing the averaged results in the Figure below:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%201%2C%20Figure%202.png" width="750"/>
</p>

<p align="center"><b>Figure 2: Averaged Top-3 Accuracy Results by Model.</b></p>

## **Task 2: Impact of Gaussian Noise on Model Performance**

To evaluate each model's robustness to image corruption, we introduced Gaussian noise to the probe images and re-evaluated performance across all indexing methods and distance metrics. Table 3 summarizes the Top-3 accuracy before and after applying noise, along with the absolute accuracy drop (Δ%):

<div align="center">

| Model         | Method      | Metric    | Clean Accuracy | Noisy Accuracy | Drop (Δ%) |
| ------------- | ----------- | --------- | -------------- | -------------- | --------- |
| casia-webface | bruteforcee | Cosine    | 13.61%     | 2.00%      | -11.61%    |
| casia-webface | bruteforce  | Euclidean | 13.61%     | 2.00%      | -11.61%    |
| casia-webface | hnsw        | Cosine    | 13.51%     | 2.10%      | -11.41%    |
| casia-webface | hnsw        | Euclidean | 13.51%     | 2.20%      | -11.31%    |
| casia-webface | lsh         | Cosine    | 6.51%      | 1.90%      | -4.61%     |
| casia-webface | lsh         | Euclidean | 6.51%      | 1.20%      | -5.31%     |
| vggface2      | bruteforce  | Cosine    | 61.96%     | 40.44%     | -21.52%    |
| vggface2      | bruteforce  | Euclidean | 61.96%     | 39.94%     | -22.02%    |
| vggface2      | hnsw        | Cosine    | 61.76%     | 39.94%     | -21.82%    |
| vggface2      | hnsw        | Euclidean | 61.76%     | 39.64%     | -22.12%    |
| vggface2      | lsh         | Cosine    | 46.35%     | 26.33%     | -20.02%    |
| vggface2      | lsh         | Euclidean | 46.35%     | 26.23%     | -20.12%    |

</div>

<p align="center"><b>Table 3: Top-3 Accuracy (Clean vs. Noisy) with Gaussian Noise.</b></p>

Despite the performance degradation caused by Gaussian noise, `vggface2` consistently outperforms `casia-webface` across all conditions. Even under the worst-case configuration (`lsh` with Euclidean distance), `vggface2` maintains a Top-3 accuracy of 26.23%, which is more than four times higher than CASIA-WebFace's best result under noise (2.20% with `hnsw` with Euclidean distance). Although `vggface2` suffers a larger absolute accuracy decline (\~20–22%), this drop is relative to a much higher starting performance. CASIA-WebFace drops by a smaller absolute margin (\~11–12%), but its final accuracy levels are critically low (below 2.5% in most cases). This suggests that `vggface2`'s embedding space retains useful identity information under distortion, while `casia-webface`'s does not. The figure shown below makes clear just how vast the differences are when comparing the results in both clean and noisy environments:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%202%2C%20Figure%202.png" width="750"/>

<p align="center"><b>Figure 3: Top-3 Accuracy Drops by Model.</b></p>

All methods experience degradation under noise, but LSH proves to be the most vulnerable, particularly for `casia-webface`. The hashing approximation amplifies the effects of noise, further compressing already fragile representations. By contrast, BruteForce and HNSW maintain relatively stable results, indicating their suitability for noisy image search applications. In noisy environments, `vggface2` demonstrates superior robustness and utility, continuing to yield useful results despite significant corruption. The `casia-webface` model, by contrast, exhibits near-total performance collapse. These findings further reinforce `vggface2` as the preferred embedding model for the IronClad system, particularly in settings with imperfect or degraded imagery. The full results with Gaussian Noise present are shown below:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%202%2C%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 4: Clean vs. Noisy Top-3 Accuracy Results.</b></p>

## **Task 3: Impact of Resizing on Model Performance**

To evaluate the robustness of each embedding model to input size variation, we adjusted the resolution of probe images to *120×120* and *200×200*, deviating from the default *160×160* baseline. Table 4 summarizes the Top-3 accuracy for each setting, while Table 5 shows the change in accuracy (Δ%) relative to the baseline:

<div align="center">

| Model         | Method     | Metric    | 120×120 Acc. | 200×200 Acc.|
| ------------- | ---------- | --------- | ------------ | ----------- |
| casia-webface | bruteforce | Cosine    | 3.00%    | 10.71%   |
| casia-webface | bruteforce | Euclidean | 3.00%    | 10.71%   |
| casia-webface | hnsw       | Cosine    | 3.00%    | 10.51%   |
| casia-webface | hnsw       | Euclidean | 3.00%    | 10.51%   |
| casia-webface | lsh        | Cosine    | 1.60%    | 3.80%    |
| casia-webface | lsh        | Euclidean | 1.60%    | 3.80%    |
| vggface2      | bruteforce | Cosine    | 5.21%    | 65.97%   |
| vggface2      | bruteforce | Euclidean | 5.21%    | 65.97%   |
| vggface2      | hnsw       | Cosine    | 5.11%    | 65.77%   |
| vggface2      | hnsw       | Euclidean | 5.11%    | 65.77%   |
| vggface2      | lsh        | Cosine    | 3.00%    | 47.05%   |
| vggface2      | lsh        | Euclidean | 3.00%    | 47.05%   |

</div>

<p align="center"><b>Table 4: Top-3 Accuracy at Different Image Sizes.</b></p>

<div align="center">

| Model         | Method     | Metric    | Δ120  | Δ200 |
| ------------- | ---------- | --------- | ------- | ------ |
| casia-webface | bruteforce | Cosine    | -10.61% | -2.90% |
| casia-webface | bruteforce | Euclidean | -10.61% | -2.90% |
| casia-webface | hnsw       | Cosine    | -10.51% | -3.00% |
| casia-webface | hnsw       | Euclidean | -10.51% | -3.00% |
| casia-webface | lsh        | Cosine    | -4.91%  | -2.71% |
| casia-webface | lsh        | Euclidean | -4.91%  | -2.71% |
| vggface2      | bruteforce | Cosine    | -56.75% | +4.01% |
| vggface2      | bruteforce | Euclidean | -56.75% | +4.01% |
| vggface2      | hnsw       | Cosine    | -56.65% | +4.01% |
| vggface2      | hnsw       | Euclidean | -56.65% | +4.01% |
| vggface2      | lsh        | Cosine    | -43.35% | +0.70% |
| vggface2      | lsh        | Euclidean | -43.35% | +0.70% |

</div>

<p align="center"><b>Table 5: Accuracy Difference (Δ%) From 160×160 Baseline.</b></p>

All models experienced a significant performance drop at the smaller size. For instance, the `casia-webface` accuracy fell by approximately −10.5% for `bruceforce` and `hnsw`, and about −4.9% for `lsh`. Comparatively, `vggface2` suffered even greater declines, about −56% to −57% across `bruteforce` and `hnsw`, due to its higher baseline. This suggests its embeddings are optimized for standard-sized inputs and lose discriminability when undersized. Interestingly, `vggface2` performance not only recovered but improved at 200×200, with Top-3 accuracy increasing by +4.01% in several configurations. This indicates the model benefits from higher input resolution, likely due to its deeper architecture and capacity to leverage fine-grained facial details. Similar to Task 2, the differences are best observed when visually represented, as shown in Figure 5:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%203%2C%20Figure%202.png" width="750"/>

<p align="center"><b>Figure 5: Top-3 Accuracy Drops by Model.</b></p>

In contrast, `casia-webface` showed only partial recovery, still remaining slightly below baseline accuracy. This suggests the model is more brittle with regard to input scale, and does not generalize well to larger images. The `lsh` index continues to exhibit more stability in accuracy deltas (smaller swings) but remains the least accurate overall. By comparison, both `bruteforce` and `hnsw` offer the best performance across both sizes for both models, but are also more sensitive to resolution changes, especially for `vggface2`. Throughout tasks 1-3, it's clear that `vggface2` is the more robust and scale-adaptive model, particularly benefiting from higher-resolution inputs. Despite sharp degradation at smaller sizes, it is the only model to surpass baseline performance when upsampled. The `casia-webface` model in contrast, underperforms at both resolutions and fails to adapt meaningfully to changes in input size. This further validates VGGFace2 as the most performant and reliable embedding backbone for the IronClad system. The improvement shown by `vggface2` for both image resizing options are best exemplified below:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%203%2C%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 6: Resized Images Top-3 Accuracy Results.</b></p>


## **Task 4 – Impact of Brightness of Model Performance**

To assess robustness to lighting variation, we evaluated model performance under two brightness conditions. For the first, we set α = 0.5, to simulate a darker environment, and afterwards made α = 1.5, to simulate a brighter one. Each test set was adjusted by multiplying pixel intensities by the respective α value. We then measured Top-3 accuracy across both models using all six method-metric combinations. Tables 6 and 7 seen below capture the results for both a lower and higher alpha value (compare to the default value of α = 0.1):

<div align="center">

| Model         | Method     | Metric    | Top-3 Accuracy | Correct Predictions |
| ------------- | ---------- | --------- | -------------- | ------------------- |
| casia-webface | bruteforce | Cosine    | 3.20%          | 32 / 999            |
| casia-webface | bruteforce | Euclidean | 3.20%          | 32 / 999            |
| casia-webface | hnsw       | Cosine    | 3.00%          | 30 / 999            |
| casia-webface | hnsw       | Euclidean | 3.00%          | 30 / 999            |
| casia-webface | lsh        | Cosine    | 1.10%          | 11 / 999            |
| casia-webface | lsh        | Euclidean | 1.10%          | 11 / 999            |
| vggface2      | bruteforce | Cosine    | 24.32%         | 243 / 999           |
| vggface2      | bruteforce | Euclidean | 24.32%         | 243 / 999           |
| vggface2      | hnsw       | Cosine    | 24.22%         | 242 / 999           |
| vggface2      | hnsw       | Euclidean | 24.22%         | 242 / 999           |
| vggface2      | lsh        | Cosine    | 15.92%         | 159 / 999           |
| vggface2      | lsh        | Euclidean | 15.92%         | 159 / 999           |

</div>

<p align="center"><b>Table 6: Performance at α = 0.5 (Darker Images).</b></p>

Under darker lighting conditions, `vggface2` remains consistently more accurate than `casia-webface`, maintaining a Top-3 accuracy of over 24% across `bruteforce` and `hnsw` methods. In contrast, `casia-webface` accuracy falls below 3.5% across all configurations. The `lsh` indexing method is again the most sensitive to the distortion, with `vggface2` and `lsh` dropping to \~16%, and the `casia-webface` with `lsh` combination plummets near 1%.

<div align="center">

| Model         | Method     | Metric    | Top-3 Accuracy | Correct Predictions |
| ------------- | ---------- | --------- | -------------- | ------------------- |
| casia-webface | bruteforce | Cosine    | 6.81%          | 68 / 999            |
| casia-webface | bruteforce | Euclidean | 6.81%          | 68 / 999            |
| casia-webface | hnsw       | Cosine    | 6.61%          | 66 / 999            |
| casia-webface | hnsw       | Euclidean | 6.61%          | 66 / 999            |
| casia-webface | lsh        | Cosine    | 3.10%          | 31 / 999            |
| casia-webface | lsh        | Euclidean | 3.10%          | 31 / 999            |
| vggface2      | bruteforce | Cosine    | 18.12%         | 181 / 999           |
| vggface2      | bruteforce | Euclidean | 18.12%         | 181 / 999           |
| vggface2      | hnsw       | Cosine    | 18.12%         | 181 / 999           |
| vggface2      | hnsw       | Euclidean | 18.12%         | 181 / 999           |
| vggface2      | lsh        | Cosine    | 11.41%         | 114 / 999           |
| vggface2      | lsh        | Euclidean | 11.41%         | 114 / 999           |

</div>

<p align="center"><b>Table 7: Performance at α = 1.5 (Brighter Images).</b></p>

Brighter inputs led to moderate degradation for both models, but the drop was less severe than in the darker case. `vggface2` continues to outperform with \~18% Top-3 accuracy using `bruteforce` and `hnsw`. The `casia-webface` model improves slightly to \~6.8% with `bruteforce`, but `lsh` again lags significantly, producing only 3.1% accuracy. It should be noted that `vggface2` is substantially more robust to brightness variation, outperforming `casia-webface` in both darker and brighter settings by large margins. The `hnsw` and `bruteforce` methods are the most consistent under lighting changes, especially with `vggface2`. The `lsh` index continues to underperform across all brightness adjustments, confirming its sensitivity to feature perturbations. Darker conditions (α = 0.5) degraded performance more heavily than brighter ones (α = 1.5), particularly for the lower-capacity `casia-webface` model. The differences can be observed in the accompanying table and figure below:

<div align="center">

| Model         | Method     | Metric    | Δ@0.5  | Δ@1.5  |
| ------------- | ---------- | --------- | ------ | ------ |
| casia-webface | bruteforce | Cosine    | -10.41% | -6.80%  |
| casia-webface | bruteforce | Euclidean | -10.41% | -6.80%  |
| casia-webface | hnsw       | Cosine    | -10.51% | -6.90%  |
| casia-webface | hnsw       | Euclidean | -10.51% | -6.90%  |
| casia-webface | lsh        | Cosine    | -5.41%  | -3.41%  |
| casia-webface | lsh        | Euclidean | -5.41%  | -3.41%  |
| vggface2      | bruteforce | Cosine    | -37.64% | -43.84% |
| vggface2      | bruteforce | Euclidean | -37.64% | -43.84% |
| vggface2      | hnsw       | Cosine    | -37.54% | -43.64% |
| vggface2      | hnsw       | Euclidean | -37.54% | -43.64% |
| vggface2      | lsh        | Cosine    | -30.43% | -34.94% |
| vggface2      | lsh        | Euclidean | -30.43% | -34.94% |

</div>

<p align="center"><b>Table 8: Accuracy Difference (Δ%) From α = 1.0 Baseline.</b></p>

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%204%2C%20Figure%202.png" alt="Figure 8: Top-3 Accuracy Drops by Model." width="750"/>

<p align="center"><b>Figure 7: Top-3 Accuracy Drops by Model.</b></p>

Figure 8 summarizes the full Top-3 accuracy results for our model/method/metric combinations given various brightness (α) combinations. As we saw with Task 3 (Gaussian noise fluctations), `vggface2` is still heads and shoulders more robust, practical, and better performing than `casia-webface`. It's worth examining if that trend will hold up once we analyze Task 5:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%204%2C%20Figure%201.png" width="750"/>
</p>

<p align="center"><b>Figure 8: Brightness Variation Top-3 Accuracy Results.</b></p>

## **Task 5 – Determining the Optimal Value of *k* (Top-*N* Candidates)**

In this task, we evaluate the effect of varying *k*, the number of candidate identities returned (Top-*k* matches), and determine the optimal value of *N = k* for our system. The analysis considers accuracy trends across Top-1, Top-3, and Top-5 retrievals for both models (`casia-webface` and `vggface2`) using all search method and metric combinations. The table below summarizes how Top-*k* accuracy changes as *k* increases from 1 to 5. We report the raw accuracy at each level, as well as the incremental improvement (Δ) between steps. It should be noted that going forward, *N* and *k* are interchangeable.

<div align="center">

| Model         | Method     | Metric    | Top-1 Accuracy | Correct Predictions |
| ------------- | ---------- | --------- | -------------- | ------------------- |
| casia-webface | bruteforce | Cosine    | 9.31%          | 93 / 999            |
| casia-webface | bruteforce | Euclidean | 9.31%          | 93 / 999            |
| casia-webface | hnsw       | Cosine    | 9.11%          | 91 / 999            |
| casia-webface | hnsw       | Euclidean | 9.11%          | 91 / 999            |
| casia-webface | lsh        | Cosine    | 3.30%          | 33 / 999            |
| casia-webface | lsh        | Euclidean | 3.30%          | 33 / 999            |
| vggface2      | bruteforce | Cosine    | 50.65%         | 506 / 999           |
| vggface2      | bruteforce | Euclidean | 50.65%         | 506 / 999           |
| vggface2      | hnsw       | Cosine    | 50.65%         | 506 / 999           |
| vggface2      | hnsw       | Euclidean | 50.65%         | 506 / 999           |
| vggface2      | lsh        | Cosine    | 33.33%         | 333 / 999           |
| vggface2      | lsh        | Euclidean | 33.33%         | 333 / 999           |

</div>

<p align="center"><b>Table 9: Top-1 Accuracy Table.</b></p>

<div align="center">

| Model         | Method     | Metric    | Top-5 Accuracy | Correct Predictions |
| ------------- | ---------- | --------- | -------------- | ------------------- |
| casia-webface | bruteforce | Cosine    | 16.32%         | 163 / 999           |
| casia-webface | bruteforce | Euclidean | 16.32%         | 163 / 999           |
| casia-webface | hnsw       | Cosine    | 15.62%         | 156 / 999           |
| casia-webface | hnsw       | Euclidean | 15.62%         | 156 / 999           |
| casia-webface | LSH        | Cosine    | 8.51%          | 85 / 999            |
| casia-webface | lsh        | Euclidean | 8.51%          | 85 / 999            |
| vggface2      | bruteforce | Cosine    | 66.47%         | 664 / 999           |
| vggface2      | bruteforce | Euclidean | 66.47%         | 664 / 999           |
| vggface2      | hnsw       | Cosine    | 66.37%         | 663 / 999           |
| vggface2      | hnsw       | Euclidean | 66.37%         | 663 / 999           |
| vggface2      | lsh        | Cosine    | 51.15%         | 511 / 999           |
| vggface2      | lsh        | Euclidean | 51.15%         | 511 / 999           |

</div>

<p align="center"><b>Table 10: Top-1 Accuracy Table.</b></p>

Again, `vggface2` consistently outperforms `casia-webface` across all Top-*k* levels and search methods. At Top-5, `vggface2` reaches up to 66.5% accuracy, while `casia-webface` peaks at just over 16%. For `vggface2`, the largest gains occur between Top-1 and Top-3 (up to +13%), with additional improvement from Top-3 to Top-5 (+4–5%). This justifies returning up to 5 candidates to users in a real system. In contrast, `casia-webface` sees smaller incremental gains, with performance increasing only \~2–4% between each step. Accuracy quickly plateaus after Top-3, suggesting limited benefit from larger *N*. The figure below summarizes that point well:

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%205%2C%20Figure%202.png" alt="Figure 9: Top-1 and Top-5 Accuracy Drops by Model." width="750"/>
  
<p align="center"><b>Figure 9: Top-1 and Top-5 Accuracy Improvements by Model.</b></p>

<div align="center">

| Model         | Search Method | Metric    | Top-1 Acc (k=1)  | Top-3 Acc (k=3)  | Δ (1→3) | Top-5 Acc (k=5)  | Δ (3→5) |
| ------------- | ------------- | --------- | ---------------- | ---------------- | --------- | ---------------- | --------- |
| CASIA-WebFace | BruteForce    | Cosine    | 9.31% (93/999)   | 13.61% (136/999) | +4.30%    | 16.32% (163/999) | +2.71%    |
| CASIA-WebFace | BruteForce    | Euclidean | 9.31% (93/999)   | 13.61% (136/999) | +4.30%    | 16.32% (163/999) | +2.71%    |
| CASIA-WebFace | HNSW          | Cosine    | 9.11% (91/999)   | 13.51% (135/999) | +4.40%    | 15.62% (156/999) | +2.11%    |
| CASIA-WebFace | HNSW          | Euclidean | 9.11% (91/999)   | 13.51% (135/999) | +4.40%    | 15.62% (156/999) | +2.11%    |
| CASIA-WebFace | LSH           | Cosine    | 3.30% (33/999)   | 6.51% (65/999)   | +3.21%    | 8.51% (85/999)   | +2.00%    |
| CASIA-WebFace | LSH           | Euclidean | 3.30% (33/999)   | 6.51% (65/999)   | +3.21%    | 8.51% (85/999)   | +2.00%    |
| VGGFace2      | BruteForce    | Cosine    | 50.65% (506/999) | 61.96% (619/999) | +11.31%   | 66.47% (664/999) | +4.51%    |
| VGGFace2      | BruteForce    | Euclidean | 50.65% (506/999) | 61.96% (619/999) | +11.31%   | 66.47% (664/999) | +4.51%    |
| VGGFace2      | HNSW          | Cosine    | 50.65% (506/999) | 61.76% (617/999) | +11.11%   | 66.37% (663/999) | +4.61%    |
| VGGFace2      | HNSW          | Euclidean | 50.65% (506/999) | 61.76% (617/999) | +11.11%   | 66.37% (663/999) | +4.61%    |
| VGGFace2      | LSH           | Cosine    | 33.33% (333/999) | 46.35% (463/999) | +13.02%   | 51.15% (511/999) | +4.80%    |
| VGGFace2      | LSH           | Euclidean | 33.33% (333/999) | 46.35% (463/999) | +13.02%   | 51.15% (511/999) | +4.80%    |

</div>

<p align="center"><b>Table 11: Accuracy Difference (Δ%) From *N* = 3 Baseline.</b></p>

Choosing the ideal value of *N* (Top-*k*) involves balancing identification accuracy against interface usability and user effort. While larger *N* increases the likelihood of including the correct identity, it also imposes cognitive overhead on the user. The `vggface2` model benefits from returning more candidates (up to 5), which allows the user to correctly identify the probe image in two-thirds of cases. Conversely, `casia-webface` performs poorly overall, and performance gains diminish beyond Top-3, making *N* = 3 the most efficient choice Overall, these findings demonstrate that the optimal *N* depends on both model capability and task context, with more accurate models benefiting from higher *N* values. The final figure shown below helps illustrate which value of *N* is more compatible for our models, methods, and metrics (including all manner of combinations).

<p align="center">
  <img src="external_plots/Assignment%206%20-%20Task%205%2C%20Figure%201.png" alt="Figure 9: Top-1 vs. Top-5 Accuracy Results." width="750"/>
</p>

<p align="center"><b>Figure 10: Top-1 vs. Top-5 Accuracy Results.</b></p>