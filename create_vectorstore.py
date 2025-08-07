import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS

# ----- Settings -----
DATA_PATH = os.path.join("data", "clinicalTrials.csv")      # Path to your CSV
FAISS_INDEX_PATH = "vectorstore"                            # Folder to save FAISS index
SAMPLE_SIZE = 3000                                          # Use a subset for fast dev (None for all rows)
BATCH_SIZE = 128                                            # Adjust for your RAM/CPU
TRUNCATE_LEN = 300                                          # Truncate to speed up
DEVICE = "cpu"                                              # Use 'cuda' if you have a supported GPU

# ----- Field Names -----
TEXT_FIELDS = ['Study Title', 'Conditions', 'Interventions', 'Brief Summary']

def truncate(text, max_length=TRUNCATE_LEN):
    return str(text)[:max_length]

def main():
    print("Loading CSV data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Original rows: {len(df)}")

    # For development, sample a subset
    if SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"Using random sample: {SAMPLE_SIZE} rows")

    # Truncate text fields for faster embedding
    for col in TEXT_FIELDS:
        df[col] = df[col].fillna("").apply(truncate)

    # Combine fields into one text per doc
    df["combined_text"] = df[TEXT_FIELDS].agg(" ".join, axis=1)
    docs = df["combined_text"].tolist()
    print(f"Preparing {len(docs)} documents for embedding.")

    # Create Document objects for FAISS (retains metadata)
    from langchain_core.documents import Document
    documents = []
    for _, row in df.iterrows():
        documents.append(Document(page_content=row["combined_text"], metadata=row.to_dict()))

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": BATCH_SIZE},  # No show_progress_bar here!
        model_kwargs={"device": DEVICE}
    )

    print(f"Computing embeddings in batches of {BATCH_SIZE}...")
    embeddings = []
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding batched docs"):
        batch = docs[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")

    print("Building FAISS vectorstore (pairing text with embeddings)...")
    text_embedding_pairs = list(zip([doc.page_content for doc in documents], embeddings))
    metadatas = [doc.metadata for doc in documents]

    # Build and save
    vectorstore = FAISS.from_embeddings(text_embedding_pairs, embedding_model, metadatas=metadatas)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Vectorstore saved to '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    main()
