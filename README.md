# IronClad Face Retrieval

A face identification service. It embeds face images with pretrained FaceNet models, stores the embeddings in FAISS indexes, and returns the closest identities for a new image through a Flask API. Two case studies measure how the embedding model, index type, distance metric, gallery size, and image quality affect retrieval accuracy.

## Results

| Finding | Result |
|---|---|
| Embedding model | FaceNet pretrained on VGGFace2 reached 50.65 percent top 1 accuracy, against 9.31 percent for the CASIA-WebFace version |
| Index type | HNSW matched brute force search almost exactly (61.76 versus 61.96 percent top 3), while LSH fell to 46.35 percent |
| Distance metric | Cosine and Euclidean distance gave identical results in every brute force and HNSW test |
| Gallery size | Raising the images per identity from one to three lifted VGGFace2 top 3 accuracy from 58.26 to 62.86 percent, with little gain beyond that |

The first [case study](ironclad/CaseStudy.md) also tests robustness to Gaussian noise, resizing, and brightness changes and chooses the number of candidates to return. The [second](ironclad/CaseStudy_2.md) adds MRR, mAP, and recall at K, and compares gallery sizes and index types.

<p align="center">
  <img src="ironclad/external_plots_m/Task%203%20-%20Plot%201.png" width="600"/>
</p>

## How it works

1. **Preprocessing.** Images are resized and normalized for the embedding model.
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
