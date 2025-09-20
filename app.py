# app.py
import streamlit as st
from pathlib import Path
import uuid
import json
import time
from src.pipeline import image_to_story

# Setup directories
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "images"
OUT_DIR = ROOT / "outputs" / "stories"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Image → Story", layout="centered")
st.title("Image → Story (BLIP + LLM)")

st.markdown("Upload an image, BLIP will create a caption, then a language model will write a short story from that caption.")

uploaded_file = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

tone = st.selectbox("Tone / Style", options=["creative", "mystery", "poetic", "dramatic", "child-friendly"])
approx_words = st.slider("Approx. story length (words)", min_value=50, max_value=400, value=180, step=10)

if uploaded_file is not None:
    # Save uploaded file
    unique_name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.{uploaded_file.name.split('.')[-1]}"
    save_path = DATA_DIR / unique_name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(str(save_path), caption="Uploaded image", use_column_width=True)

    if st.button("Generate Caption & Story"):
        try:
            with st.spinner("Generating caption and story..."):
                caption, story = image_to_story(str(save_path), tone=tone, approx_words=approx_words)

            st.subheader("Caption")
            st.write(caption)

            st.subheader("Story")
            st.write(story)

            # Save output JSON
            out = {
                "image_filename": str(save_path.name),
                "caption": caption,
                "tone": tone,
                "approx_words": approx_words,
                "story": story,
                "timestamp": int(time.time())
            }
            out_filename = OUT_DIR / f"{save_path.stem}.json"
            with open(out_filename, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

            st.success(f"Done — output saved to {out_filename}")
        except Exception as e:
            st.error(f"Error: {e}")
            raise
else:
    st.info("Upload an image to start. Click 'Generate Caption & Story' after upload.")
