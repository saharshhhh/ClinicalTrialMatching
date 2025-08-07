import os
from flask import Flask, render_template, request, jsonify
from retriever import load_retriever, load_dataframe
from utils import get_trial_metadata, build_prompt
from llm import load_local_llm  # Adjust to use your LLM loader (local or HF endpoint)

# Initialize Flask app
app = Flask(__name__)

# Global variables (load once at startup)
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "vectorstore")
DATA_PATH = os.getenv("DATA_PATH", "data/clinicalTrials.csv")

# Load retriever, df, and LLM once to save time on requests
print("Loading retriever and dataframe...")
retriever = load_retriever()
df = load_dataframe()
print("Loading LLM pipeline...")
llm = load_local_llm()

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

    prompt = build_prompt(user_query, metadata)

    # Generate answer
    if hasattr(llm, "invoke"):  # HuggingFaceEndpoint (async) usage
        result = llm.invoke(prompt)
    else:  # transformers pipeline usage
        result = llm(prompt, max_new_tokens=256)[0]["generated_text"]

    return jsonify({
        "trial_title": metadata["Study Title"],
        "nct_number": metadata.get("NCT Number", "N/A"),
        "summary": result
    })


if __name__ == "__main__":
    # Run app in debug mode for development
    app.run(host="0.0.0.0", port=5000, debug=True)
