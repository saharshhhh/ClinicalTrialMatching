import google.generativeai as genai
from config import GEMINI_API_KEY
import os

def load_summarization_model():
    """
    Returns a function that interacts with Google Gemini API to summarize text.
    mimics the calling signature expected by app.py (text input), 
    but we will update app.py to match this new signature if needed, 
    or keep it simple.
    """
    
    if not GEMINI_API_KEY:
        print("WARNING: GEMINI_API_KEY not found in environment variables.")
        return lambda text, **kwargs: "Error: GEMINI_API_KEY not configured."

    # Configure the API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Use gemini-pro as a fallback
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Define the system prompt / persona here
    # We can prepend this to the message
    SYSTEM_INSTRUCTION = (
        "You are an expert medical communicator. Your goal is to explain clinical trial "
        "details to a layperson (a patient or family member) who has no medical background. "
        "Use simple, clear language. Avoid jargon where possible, or explain it if necessary. "
        "Focus on the purpose, the treatments, the process, and the potential outcomes. "
        "Be empathetic but objective."
    )

    def summarize(text, **kwargs):
        """
        Generates a patient-friendly summary using Gemini.
        Ignores HF pipeline kwargs like max_length/do_sample.
        """
        try:
            # Construct the full prompt
            prompt = f"{SYSTEM_INSTRUCTION}\n\nHere is the clinical trial information:\n{text}\n\nProvide a clear, patient-friendly summary politely."
            
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"Error generating summary: {str(e)}"

    return summarize
