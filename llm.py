from transformers import pipeline

# def load_text_generation_model():
#     """Load text generation LLM (for chat-like, dialog responses only)."""
#     return pipeline("text-generation", model="distilgpt2")

def load_summarization_model():
    return pipeline("summarization", model="weijiahaha/t5-small-summarization")


