from transformers import pipeline

def load_text_generation_model():
    return pipeline("text-generation", model="distilgpt2")

def load_summarization_model():
    return pipeline("summarization", model="google/flan-t5-large")

# If you want backward compatibility with prev. code:
def load_local_llm(task="text-generation"):
    if task == "text-generation":
        return load_text_generation_model()
    elif task == "summarization":
        return load_summarization_model()
    else:
        raise ValueError("Invalid task type!")
