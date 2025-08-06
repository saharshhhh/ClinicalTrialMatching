import os
from config import FAISS_INDEX_PATH
from retriever import load_retriever, load_dataframe
from utils import get_trial_metadata, build_prompt

# Choose one accordingly:
from llm import load_local_llm
# from llm import load_hf_llm   # Uncomment if using HF API instead

def main():
    # Check if vectorstore index exists, else create it
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        print("Vectorstore index not found. Please run 'create_vectorstore.py' first.")
        return

    retriever = load_retriever()
    df = load_dataframe()
    llm = load_local_llm()  # or load_hf_llm()

    print("\nClinical Trials Chatbot is ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        docs = retriever.get_relevant_documents(query)
        if not docs:
            print("No matching trials found. Please try rephrasing your query.\n")
            continue

        best_doc = docs[0]
        metadata = get_trial_metadata(best_doc.page_content, df)
        if not metadata:
            print("A matching trial was found but details could not be extracted.\n")
            continue

        prompt = build_prompt(query, metadata)

        # Local pipeline usage
        if hasattr(llm, "invoke"):  # HF Endpoint usage
            response = llm.invoke(prompt)
        else:  # transformers pipeline usage
            response = llm(prompt, max_new_tokens=256)[0]['generated_text']

        print(f"\nTop matching trial: {metadata['Study Title']} (NCT: {metadata['NCT Number']})")
        print("Generated summary:")
        print(response, "\n")


if __name__ == "__main__":
    main()
