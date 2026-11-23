"""Usage: python scripts/compare_preprocessing.py [path/to/multi_image_identities]

Compare IronClad's original preprocessing with the standard FaceNet pipeline.

original: resize the whole 250x250 photo to 160x160, ImageNet mean/std normalization
           (what ironclad/modules/extraction/preprocessing.py does)
mtcnn:    detect and crop the face with MTCNN at 160x160 with a 20 px margin, then
           FaceNet's fixed standardization (x - 127.5) / 128 (facenet-pytorch's intended input)

Gallery: every image in multi_image_gallery. Probes: one image per identity in probe/.
Search: exact cosine similarity (L2-normalized embeddings, inner product), as IronClad's
brute-force FAISS index does. Reports top-1 / top-3 / top-5 identity accuracy and MRR.
"""
import glob, os, json, sys, time
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization

torch.set_num_threads(os.cpu_count())
ROOT = sys.argv[1] if len(sys.argv) > 1 else "storage/multi_image_identities"
gallery = sorted(glob.glob(f"{ROOT}/multi_image_gallery/*/*.jpg"))
probes = sorted(glob.glob(f"{ROOT}/probe/*/*.jpg"))
ident = lambda p: p.split("/")[-2]

orig_tf = transforms.Compose([transforms.Resize((160, 160)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
center_tf = transforms.Compose([transforms.CenterCrop(160), transforms.ToTensor(), fixed_image_standardization])
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device="cpu")

def load_original(paths):
    return torch.stack([orig_tf(Image.open(p).convert("RGB")) for p in paths])

misses = 0
def load_mtcnn(paths):
    global misses
    out = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        face = mtcnn(img)
        if face is None:            # no face found: fall back to a centre crop
            misses += 1
            face = center_tf(img)
        out.append(face)
    return torch.stack(out)

def embed(model, paths, loader, bs=64):
    vecs = []
    for i in range(0, len(paths), bs):
        with torch.no_grad():
            vecs.append(model(loader(paths[i:i + bs])).numpy())
    v = np.concatenate(vecs)
    return v / np.linalg.norm(v, axis=1, keepdims=True)

def score(g, p):
    g_ids = np.array([ident(x) for x in gallery]); p_ids = [ident(x) for x in probes]
    sims = p @ g.T
    hits = {1: 0, 3: 0, 5: 0}; rr = 0.0
    for i, pid in enumerate(p_ids):
        order = np.argsort(-sims[i]); ranked = []
        for j in order:                      # rank identities by their best gallery match
            if g_ids[j] not in ranked:
                ranked.append(g_ids[j])
            if len(ranked) == 20:
                break
        for k in hits:
            hits[k] += pid in ranked[:k]
        rr += 1.0 / (ranked.index(pid) + 1) if pid in ranked else 0.0
    n = len(p_ids)
    return {f"top{k}": round(100 * v / n, 2) for k, v in hits.items()} | {"mrr": round(rr / n, 4)}

results = {}
for weights in ["vggface2", "casia-webface"]:
    model = InceptionResnetV1(pretrained=weights).eval()
    for name, loader in [("original", load_original), ("mtcnn", load_mtcnn)]:
        t = time.time(); misses = 0
        res = score(embed(model, gallery, loader), embed(model, probes, loader))
        res["no_face_fallbacks"] = misses if name == "mtcnn" else None
        results[f"{weights}/{name}"] = res
        print(f"{weights:14s} {name:9s} {res}  ({time.time() - t:.0f}s)", flush=True)
json.dump(results, open("preprocessing_results.json", "w"), indent=2) # written to the working directory
print(f"gallery {len(gallery)} images / {len(set(map(ident, gallery)))} identities, probes {len(probes)}")
