import os
from flask import Flask, render_template, request, jsonify

from retriever import load_retriever, load_dataframe
from utils import get_trial_metadata

# Import the correct model loader
from llm import load_summarization_model

app = Flask(__name__)

# Set paths
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore")
DATA_PATH = os.getenv("DATA_PATH", "data/clinicalTrials.csv")

print("Loading retriever and dataframe...")
retriever = load_retriever()
df = load_dataframe()
print("Loading summarization model...")
summarizer = load_summarization_model()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_query = request.json.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    docs = retriever.get_relevant_documents(user_query)
    if not docs:
        return jsonify({"error": "No matching trials found."}), 404

    best_doc = docs[0]
    metadata = get_trial_metadata(best_doc.page_content, df)
    if not metadata:
        return jsonify({"error": "Matching trial found but details extraction failed."}), 500

    # Only summarize the relevant clinical content
    to_summarize = (
        f"Study Design: {metadata.get('Study Design', '')}\n"
        f"Interventions: {metadata.get('Interventions', '')}\n"
        f"Brief Summary: {metadata.get('Brief Summary', '')}"
    )

    # --- PROMPT ENHANCEMENT ---
    # This tells the summarizer to produce an expanded, plain-language summary
    patient_friendly_prompt = (
        "Rewrite the following clinical trial details so that an average person "
        "with no deep medical knowledge can understand it. "
        "Provide a clear, detailed, and easy-to-read explanation of the study’s purpose, "
        "who it is for, what treatments are being tested, how it is conducted, and "
        "what outcomes are expected:\n\n"
        f"{to_summarize}"
    )

    # Generate the longer patient-friendly summary
    summary = summarizer(
        patient_friendly_prompt,
        max_length=512,   # Increased for more elaboration
        min_length=120,    # Increased minimum to ensure good detail
        do_sample=False
    )[0]["summary_text"]

    return jsonify({
        "trial_title": metadata.get("Study Title", ""),
        "nct_number": metadata.get("NCT Number", "N/A"),
        "summary": summary
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
