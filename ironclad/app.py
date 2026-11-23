"""
Flask app for processing images.

This script provides two endpoints:
1. /identify: Processes an image and returns the top-k identities.
2. /add: Adds a provided image to the gallery with an associated name.

Usage:
    Run the app with: python app.py
    Sample curl command for /identify:
        curl -X POST -F "image=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
    Sample curl command for /add:
        curl -X POST -F "image=@/path/to/image.jpg" -F "name=Firstname_Lastname" http://localhost:5000/add
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__))) # Add current directory to system path for module resolution.

import os.path
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify

from modules.extraction.embedding import Embedding # Internal modules for processing and indexing.
from modules.extraction.preprocessing import Preprocessing
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.search import FaissSearch

app = Flask(__name__) # Initialize the Flask app.

DEFAULT_K = '3' # Default number of identities to return.
MODEL = 'vggface2' # Pretrained model to use.
INDEX = 'hnsw' # Index type.
SIMILARITY_MEASURE = 'cosine' # Similarity measure for search.
SAVE_INDEX = False # Whether to persist the index to disk.
INDEX_PATH = 'scripts/faiss_hnsw_index.pkl' # Path to store/load the index.

preprocessor = Preprocessing() # Initialize preprocessing and embedding model.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Embedding(pretrained=MODEL, device=device)

if SAVE_INDEX and os.path.exists(INDEX_PATH): # Load or initialize the FAISS index.
    index = FaissHNSW.load(INDEX_PATH)
else:
    index = FaissHNSW(dim=512, metric=SIMILARITY_MEASURE)

search = FaissSearch(index, metric=SIMILARITY_MEASURE) # Initialize the search wrapper.

@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Parameters:
        image (FileStorage): The input image to identify.
        k (int, optional): Number of top matches to return. Defaults to 3.

    Returns:
        Response (JSON): {
            "message": str,
            "ranked identities": List[str]
        }
        or error message with HTTP 400/500.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        k = int(request.form.get('k', DEFAULT_K)) # Parse `k` parameter with fallback.
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    try:
        image = np.array(Image.open(file)) # Attempt to open and convert image to NumPy.
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    image_tensor = preprocessor.process(Image.fromarray(image)) # Preprocess and extract embedding from the image.
    embedding = model.encode(image_tensor)

    distances, indices, metadatas = search.search(embedding, k) # Perform similarity search.

    print("distances: ", distances)
    print("indices: ", indices)
    print("metadatas: ", metadatas)

    identities = [name for name in metadatas[0] if name is not None] # Extract the list of identities found.

    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": identities
    }), 200

@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Parameters:
        image (FileStorage): The image to add to the gallery.
        name (str): Identity label to associate with the image.

    Returns:
        Response (JSON): {
            "message": str
        }
        or error message with HTTP 400/500.
    """
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    name = request.form.get('name')

    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    save_dir = os.path.join("storage", "gallery", name)

    if os.path.exists(save_dir) and os.listdir(save_dir):
        return jsonify({"Error": f"Gallery entry for '{name}' already exists"}), 400

    try:
        os.makedirs(save_dir, exist_ok=True)

        safe_filename = file.filename
        image_path = os.path.join(save_dir, safe_filename) # Define image path properly.
        file.save(image_path)

        image_tensor = preprocessor.process(Image.open(image_path))
        embedding = model.encode(image_tensor)

        if embedding is None: # Error printing statement.
            return jsonify({"Error": "Failed to encode image"}), 400

        index.add_embeddings([embedding], [name])

        if SAVE_INDEX:
            index.save(INDEX_PATH)

        return jsonify({
            "message": f"New image saved to gallery ({name}) and indexed."
        }), 200 # Confirmation of image save to gallery.

    except Exception:
        return jsonify({"Error": "Unexpected error during image processing"}), 500 # Error printing statement.
    
if __name__ == '__main__':
    app.run(debug=True)