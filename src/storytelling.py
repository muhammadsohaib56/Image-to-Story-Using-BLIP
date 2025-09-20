# src/storytelling.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def generate_story_from_caption(
    caption: str,
    tone: str = "creative",
    approx_words: int = 180,
    model: str = "gemini-1.5-flash"
) -> str:
    """
    Calls Gemini to generate a story from a caption.
    - caption: short caption string
    - tone: 'creative', 'mystery', 'poetic', etc.
    - approx_words: approximate desired word length
    - model: Gemini model name (default gemini-1.5-flash)
    """

    # Build prompt
    prompt = (
        f"Caption: \"{caption}\"\n\n"
        f"Write a {tone} story of about {approx_words} words in 2-3 short paragraphs. "
        "Do NOT list objects; weave visual details naturally into the narrative. "
        "Keep language vivid and simple."
    )

    # Call Gemini
    model_obj = genai.GenerativeModel(model)
    response = model_obj.generate_content(prompt)

    if response.candidates and response.candidates[0].content.parts:
        return response.candidates[0].content.parts[0].text.strip()
    else:
        return "No story generated. Please try again."
