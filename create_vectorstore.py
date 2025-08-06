import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_PATH, FAISS_INDEX_PATH

def create_vectorstore():
    print("Loading and prepping clinical trials dataset...")
    df = pd.read_csv(DATA_PATH)
    text_fields = ['Study Title', 'Conditions', 'Interventions', 'Brief Summary']
    df['combined_text'] = df[text_fields].astype(str).agg(' '.join, axis=1)
    loader = DataFrameLoader(df, page_content_column="combined_text")
    documents = loader.load()
    
    print(f"Loaded {len(documents)} documents.")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(documents, embedding_model)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Vectorstore saved at '{FAISS_INDEX_PATH}'.")

if __name__ == "__main__":
    create_vectorstore()
