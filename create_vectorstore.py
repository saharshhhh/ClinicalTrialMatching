import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.init import Auth
from langchain_huggingface import HuggingFaceEmbeddings
from config import DATA_PATH, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS_NAME

# ----- Settings -----
SAMPLE_SIZE = 10000  # Use a subset for fast dev (None for all rows) # TODO: revert to None
BATCH_SIZE = 128      # Adjust for your RAM/CPU
TRUNCATE_LEN = 300    # Truncate to speed up
DEVICE = "cpu"        # Use 'cuda' if you have a supported GPU

# ----- Field Names -----
TEXT_FIELDS = ['Study Title', 'Conditions', 'Interventions', 'Brief Summary']

def truncate(text, max_length=TRUNCATE_LEN):
    return str(text)[:max_length]

def create_weaviate_client():
    """Create and configure Weaviate client"""
    # Detect if we should use local or cloud connection
    # Simple heuristic: if URL contains 'localhost' or is short, assume local/custom
    # If it is a full URL, we might need to parse.
    # However, connect_to_weaviate_cloud requires cluster URL (host) and api key.
    # connect_to_local is for local instances.

    # If WEAVIATE_URL implies cloud (e.g. weaviate.network or weaviate.cloud), use connect_to_weaviate_cloud
    if "weaviate.network" in WEAVIATE_URL or "weaviate.cloud" in WEAVIATE_URL:
        # Ensure URL has https prefix
        cluster_url = WEAVIATE_URL
        if not cluster_url.startswith("https://") and not cluster_url.startswith("http://"):
            cluster_url = "https://" + cluster_url
            
        print(f"Connecting to Weaviate Cloud: {cluster_url}")
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=cluster_url, 
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None,
            # Skip verify check might be needed for some setups but usually safe to omit
        )
    else:
        # Assume local if not a clear cloud URL
        print(f"Connecting to local Weaviate: {WEAVIATE_URL}")
        # For local, often we just need port or host. connect_to_local defaults to localhost:8080.
        # If the user has a custom URL in env, we might need connect_to_custom
        # but for safety let's try connect_to_custom as it is more generic
        host = WEAVIATE_URL.replace("http://", "").replace("https://", "").split(":")[0]
        try:
            port = int(WEAVIATE_URL.split(":")[-1])
        except ValueError:
            port = 8080
            
        return weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=WEAVIATE_URL.startswith("https"),
            grpc_host=host,
            grpc_port=50051, # Default gRPC port, valid for most docker setups
            grpc_secure=WEAVIATE_URL.startswith("https"),
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
        )

def create_schema(client):
    """Create the schema for clinical trials in Weaviate"""
    # In v4, we don't need to manually check exists -> delete.
    # create_collection will fail if exists, so we check first.
    
    if client.collections.exists(WEAVIATE_CLASS_NAME):
        print(f"Deleting existing class '{WEAVIATE_CLASS_NAME}'...")
        client.collections.delete(WEAVIATE_CLASS_NAME)
    
    print(f"Creating class '{WEAVIATE_CLASS_NAME}'...")
    client.collections.create(
        name=WEAVIATE_CLASS_NAME,
        description="Clinical trials data",
        vectorizer_config=Configure.Vectorizer.none(), # We provide our own vectors
        properties=[
            Property(name="content", data_type=DataType.TEXT, description="Combined text content for search"),
            Property(name="study_title", data_type=DataType.TEXT, description="Study title"),
            Property(name="nct_number", data_type=DataType.TEXT, description="NCT number"),
            Property(name="conditions", data_type=DataType.TEXT, description="Study conditions"),
            Property(name="interventions", data_type=DataType.TEXT, description="Study interventions"),
            Property(name="brief_summary", data_type=DataType.TEXT, description="Brief summary"),
            Property(name="study_design", data_type=DataType.TEXT, description="Study design")
        ]
    )

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

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": BATCH_SIZE},
        model_kwargs={"device": DEVICE}
    )

    print(f"Computing embeddings in batches of {BATCH_SIZE}...")
    embeddings = []
    # To save memory, we could stream this, but let's stick to the list approach for now
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding batched docs"):
        batch = docs[i:i + BATCH_SIZE]
        batch_embeddings = embedding_model.embed_documents(batch)
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    print(f"Embedding shape: {embeddings.shape}")

    print("Connecting to Weaviate and uploading...")
    
    # Use context manager for auto-close
    with create_weaviate_client() as client:
        create_schema(client)
        
        collection = client.collections.get(WEAVIATE_CLASS_NAME)
        
        print("Uploading documents to Weaviate...")
        with collection.batch.dynamic() as batch:
            for i, (_, row) in enumerate(df.iterrows()):
                data_obj = {
                    "content": row["combined_text"],
                    "study_title": str(row.get("Study Title", "")),
                    "nct_number": str(row.get("NCT Number", "")),
                    "conditions": str(row.get("Conditions", "")),
                    "interventions": str(row.get("Interventions", "")),
                    "brief_summary": str(row.get("Brief Summary", "")),
                    "study_design": str(row.get("Study Design", ""))
                }
                
                # Add object to batch
                batch.add_object(
                    properties=data_obj,
                    vector=embeddings[i].tolist()
                )
                
                if i > 0 and i % 1000 == 0:
                     print(f"Queued {i} documents...")

        if len(client.batch.failed_objects) > 0:
            print(f"Failed to import {len(client.batch.failed_objects)} objects")
            for failed in client.batch.failed_objects[:5]:
                print(f"Failed object: {failed}")
        
    print(f"Successfully uploaded documents to Weaviate class '{WEAVIATE_CLASS_NAME}'.")

if __name__ == "__main__":
    main()
