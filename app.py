from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import faiss
import pickle
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow frontend requests

# Load Data and Models with Error Handling
try:
    df = pd.read_csv("flask-backend\processed_medicine_data.csv")
    df["name_lower"] = df["name"].str.lower()  # Ensure case-insensitive search
    print("✅ CSV loaded successfully.")
except FileNotFoundError:
    print("❌ Error: processed_medicine_data.csv not found.")
    df = None

try:
    model = pickle.load(open("flask-backend\composition_model.pkl", "rb"))
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print("❌ Error: composition_model.pkl not found.")
    model = None

try:
    index = faiss.read_index("flask-backend\composition_index.faiss")
    print("✅ FAISS index loaded successfully.")
except FileNotFoundError:
    print("❌ Error: composition_index.faiss not found.")
    index = None


@app.route("/recommend", methods=["POST"])
def recommend_medicine():
    """
    Given a medicine name, this function returns similar alternatives.
    """
    if df is None or model is None or index is None:
        return jsonify({"error": "Server error: Missing required files."}), 500

    data = request.json
    medicine_name = data.get("medicine_name", "").strip().lower()

    print(f"🔍 Searching for medicine: {medicine_name}")  # Debug log

    # Find the medicine in the dataset
    input_row = df[df["name_lower"] == medicine_name]

    if input_row.empty:
        print("❌ Medicine not found in CSV.")
        return jsonify({"error": "Medicine not found"}), 404

    input_composition = input_row["composition"].iloc[0]

    # Convert composition to embedding
    input_embedding = model.encode([input_composition], normalize_embeddings=True)

    # Search for similar compositions
    D, I = index.search(np.array(input_embedding), 6)  # Get top 6 results

    # Get recommended medicines
    recommended_meds = df.iloc[I[0]].copy()

    # Exclude the input medicine itself
    recommended_meds = recommended_meds[recommended_meds["name_lower"] != medicine_name]

    # Rank by similarity, then by price
    recommended_meds["similarity_score"] = D[0][: len(recommended_meds)]
    recommended_meds = recommended_meds.sort_values(by=["similarity_score", "price(₹)"], ascending=[True, True])

    print(f"✅ Found {len(recommended_meds)} alternatives.")  # Debug log

    return jsonify({
        "query": medicine_name,
        "recommendations": recommended_meds[["name", "price(₹)", "manufacturer_name", "composition", "pack_size_label", "short_composition1"]].to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
