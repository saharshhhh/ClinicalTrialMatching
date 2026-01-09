import os
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.path.join("data", "clinicalTrials.csv")

# Weaviate configuration
# For local Docker: http://localhost:8080
# For Weaviate Cloud: https://your-cluster-url.weaviate.network
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "https://clinical-trials-8x8qjf8x.weaviate.network")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "YOUR_API_KEY_HERE")
WEAVIATE_CLASS_NAME = "ClinicalTrial"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
