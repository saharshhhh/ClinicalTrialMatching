import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_PATH, FAISS_INDEX_PATH

def load_retriever():
    print("Loading retriever from FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  # top 1 document
    print("Retriever loaded.")
    return retriever

def load_dataframe():
    df = pd.read_csv(DATA_PATH)
    return df.fillna("")  # Replace NaN with empty string for safety
