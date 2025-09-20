# src/captioning.py
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import os

# Globals cached so model loads only once during session
_processor = None
_model = None
_device = None

def _load_model():
    global _processor, _model, _device
    if _model is not None and _processor is not None:
        return _processor, _model, _device

    # device = 'cuda' if available else 'cpu'
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[captioning] loading BLIP model on {_device} (this may take a while)...")

    _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    _model.to(_device)
    _model.eval()
    return _processor, _model, _device

def generate_caption(image_path: str, max_length: int = 30) -> str:
    """
    Generate a single caption for the image at image_path.
    Returns the caption string.
    """
    processor, model, device = _load_model()

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=3, early_stopping=True)

    # decode tokens -> string
    caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return caption
