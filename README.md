# IronClad Face Retrieval

A face identification service. It embeds face images with pretrained FaceNet models, stores the embeddings in FAISS indexes, and returns the closest identities for a new image through a Flask API. Two case studies measure how the embedding model, index type, distance metric, gallery size, and image quality affect retrieval accuracy.

## Results

The biggest gain came from fixing the input to the embedding model. FaceNet was trained on tightly cropped faces scaled with its own standardization, but the original pipeline resized the whole photo and applied ImageNet normalization. Detecting and cropping the face with MTCNN and using FaceNet's standardization raised accuracy for both pretrained models, measured on the same 999 probes against a 2,265 image gallery with exact cosine search.

| Pretrained weights | Preprocessing | Top 1 | Top 3 | MRR |
|---|---|---|---|---|
| VGGFace2 | Original (whole image, ImageNet normalization) | 50.65% | 62.46% | 0.581 |
| VGGFace2 | **MTCNN face crop, FaceNet standardization** | **93.29%** | **94.79%** | **0.941** |
| CASIA-WebFace | Original | 9.31% | 13.71% | 0.130 |
| CASIA-WebFace | **MTCNN face crop, FaceNet standardization** | **90.89%** | **92.99%** | **0.922** |

The large gap between the two models in the original case studies came mostly from the preprocessing. With the correct input, both perform well and VGGFace2 leads by about two points. [scripts/compare_preprocessing.py](ironclad/scripts/compare_preprocessing.py) reproduces the table.

These findings from the case studies hold regardless of preprocessing.

| Finding | Result |
|---|---|
| Index type | HNSW matched brute force search almost exactly, while LSH lost substantial accuracy |
| Distance metric | Cosine and Euclidean distance gave identical results in every brute force and HNSW test |
| Gallery size | More images per identity improved accuracy, with most of the gain by three images |

The [case studies](ironclad/CaseStudy.md) were run with the original preprocessing, so their absolute accuracies are lower than the table above. The [second](ironclad/CaseStudy_2.md) adds MRR, mAP, and recall at K and compares gallery sizes and index types.

## How it works

1. **Preprocessing.** MTCNN detects and crops the face to 160 by 160 pixels and applies FaceNet's fixed standardization, with a centre crop fallback when no face is found.
2. **Embedding.** `Embedding` wraps FaceNet (InceptionResnetV1 from facenet-pytorch) to produce 512 dimensional vectors.
3. **Indexing.** Brute force, HNSW, and LSH FAISS indexes store the gallery embeddings.
4. **Search.** `FaissSearch` returns the top K identities by cosine or Euclidean distance.
5. **Serving.** `app.py` exposes `/identify` to match a probe image and `/add` to enroll a new image under a name.

## Running it

The face image galleries and probes used in the case studies were provided by the course and are not included. The API starts with an empty index, and you can enroll your own images through `/add`.

```bash
pip install -r ironclad/requirements.txt
cd ironclad
python app.py

curl -X POST -F "image=@person.jpg" -F "name=Firstname_Lastname" http://localhost:5000/add
curl -X POST -F "image=@probe.jpg" -F "k=3" http://localhost:5000/identify
```

## Built with

Python, PyTorch, facenet-pytorch, FAISS, scikit-learn, Flask

## Context

Built for EN.705.603 Creating AI-Enabled Systems at Johns Hopkins University (Summer 2025). Course data, starter code, and unit tests are not included.
