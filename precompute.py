import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer

def load_trials_for_encoding():
    trials = []
    csv_path = os.path.join(os.path.dirname(__file__), 'trials.csv')
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            trials.append({
                "title":        row.get("Brief Title") or "",
                "condition":    row.get("Conditions") or "",
                "description":  row.get("Intervention Description") or "",
            })
    return trials

print("Loading trials...")
ALL_TRIALS = load_trials_for_encoding()
MATCHING_SAMPLE_SIZE = 50000 
matching_trials = ALL_TRIALS[:MATCHING_SAMPLE_SIZE]

# Build the text strings exactly how app.py does it
trial_texts = [
    " ".join(filter(None, [
        str(t.get("title", "")),
        str(t.get("condition", "")),
        str(t.get("description", "")),
    ]))
    for t in matching_trials
]

print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print(f"Encoding {len(trial_texts)} trials (This takes time, but only happens once!)...")
# batch_size=32 is the magic number to stop your RAM from crashing
trial_vectors = model.encode(trial_texts, batch_size=32, show_progress_bar=True)

# Save to your data directory
os.makedirs('data', exist_ok=True)
save_path = os.path.join('data', 'trial_embeddings.npy')
np.save(save_path, trial_vectors)

print(f"\nSuccess! Embeddings saved to {save_path}.")
print("You can now safely run app.py.")