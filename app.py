from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import pickle
import os
import gdown

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow frontend requests

# Google Drive File IDs (Replace these with your actual IDs)
MODEL_FILE_ID = "1ywnMbaunPdn_u6WmF8UVOmOg8ZdgaTPR" 
INDEX_FILE_ID = "17MuyxuTTor2OoLJDaw0wiaEY7YJ5hyOH"

# File paths
MODEL_PATH = "composition_model.pkl"
INDEX_PATH = "composition_index.faiss"

# Function to download files from Google Drive
def download_file(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"📥 Downloading {output_path} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Download required files
download_file(MODEL_FILE_ID, MODEL_PATH)
download_file(INDEX_FILE_ID, INDEX_PATH)

# Load Data and Models

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get directory of app.py
CSV_PATH = os.path.join(BASE_DIR, "processed_medicine_data.csv")  # Adjusted path

try:
    df = pd.read_csv(CSV_PATH)
    df["name_lower"] = df["name"].str.lower()
    print("✅ CSV loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: {CSV_PATH} not found.")
    df = None



try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: Model file not found.")
    model = None

try:
    index = faiss.read_index(INDEX_PATH)
    print("✅ FAISS index loaded successfully.")
except FileNotFoundError:
    print("❌ Error: FAISS index file not found.")
    index = None

@app.route("/recommend", methods=["POST"])
def recommend_medicine():
    """Given a medicine name, return similar alternatives."""
    if df is None or model is None or index is None:
        return jsonify({"error": "Server error: Missing required files."}), 500

    data = request.json
    medicine_name = data.get("medicine_name", "").strip().lower()
    print(f"🔍 Searching for medicine: {medicine_name}")

    input_row = df[df["name_lower"] == medicine_name]
    if input_row.empty:
        return jsonify({"error": "Medicine not found"}), 404

    input_composition = input_row["composition"].iloc[0]
    input_embedding = model.encode([input_composition], normalize_embeddings=True)

    # Search for similar compositions
    D, I = index.search(np.array(input_embedding), 6)
    recommended_meds = df.iloc[I[0]].copy()
    recommended_meds = recommended_meds[recommended_meds["name_lower"] != medicine_name]
    recommended_meds["similarity_score"] = D[0][: len(recommended_meds)]
    recommended_meds = recommended_meds.sort_values(by=["similarity_score", "price(₹)"], ascending=[True, True])

    return jsonify({
        "query": medicine_name,
        "recommendations": recommended_meds[["name", "price(₹)", "manufacturer_name", "composition", "pack_size_label", "short_composition1"]].to_dict(orient="records")
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
