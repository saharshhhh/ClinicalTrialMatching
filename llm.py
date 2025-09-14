from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_summarization_model():
    # Load T5 tokenizer & model for plain-language friendly summarization
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    
    # Return a function that mimics the HF summarization pipeline output format
    def summarize(text, max_length=180, min_length=20, do_sample=False):
        # Tokenize input
        tok_input = tokenizer.batch_encode_plus(
            [text], return_tensors="pt", padding=True, truncation=True
        )
        # Generate output
        summary_ids = model.generate(
            **tok_input,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample
        )
        # Decode tokens
        decoded = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        # Match the existing .pipeline() return structure
        return [{"summary_text": decoded[0]}]

    return summarize
