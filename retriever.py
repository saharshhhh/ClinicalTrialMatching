import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_weaviate import WeaviateVectorStore
import weaviate
from weaviate.classes.init import Auth
from config import DATA_PATH, WEAVIATE_URL, WEAVIATE_API_KEY, WEAVIATE_CLASS_NAME

def get_weaviate_client():
    """Establish connection to Weaviate v4"""
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
            grpc_port=50051,
            grpc_secure=WEAVIATE_URL.startswith("https"),
            auth_credentials=Auth.api_key(WEAVIATE_API_KEY) if WEAVIATE_API_KEY else None
        )

def load_retriever():
    print("Loading retriever from Weaviate...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Weaviate v4 client
    # Note: The client is created here and passed to the vectorstore. 
    # It will remain open for the lifecycle of the application.
    client = get_weaviate_client()
    
    # Create Weaviate vectorstore using langchain-weaviate
    vectorstore = WeaviateVectorStore(
        client=client,
        index_name=WEAVIATE_CLASS_NAME,
        text_key="content",
        embedding=embedding_model,
        attributes=["study_title", "nct_number", "conditions", "interventions", "brief_summary", "study_design"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 documents
    print("Retriever loaded.")
    return retriever

def load_dataframe():
    df = pd.read_csv(DATA_PATH)
    return df.fillna("")
