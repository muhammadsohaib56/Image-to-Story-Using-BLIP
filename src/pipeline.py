# src/pipeline.py
from .captioning import generate_caption
from .storytelling import generate_story_from_caption

def image_to_story(image_path: str, tone: str = "creative", approx_words: int = 180):
    """
    Run the two-step pipeline: image -> caption, caption -> story.
    Returns (caption, story)
    """
    caption = generate_caption(image_path)
    story = generate_story_from_caption(caption, tone=tone, approx_words=approx_words)
    return caption, story
