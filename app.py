"""
Article Image Generator — Flask Web App
"""

import json
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

OUTPUT_DIR = Path("generated_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Style constants ────────────────────────────────────────────────────────────

STYLE_GUIDE = (
    "hand-drawn educational diagram, black ink on cream/beige paper background, "
    "sketch style, simple cartoon characters with minimal detail, clean outlines, "
    "annotated with clear bold labels and arrows, thought bubbles and speech bubbles, "
    "comparison/analogy layout, friendly and approachable visual metaphor, "
    "no color fills except occasional light pastel highlights for emphasis, "
    'bottom caption in italic font, "wait but why" or "sketchplanations" illustration style'
)

CLAUDE_SYSTEM = """You are a visual communication expert who designs hand-drawn educational diagrams
in the style of 'Wait But Why' or Tim Urban's sketches — simple, witty, and insightful.

Your job is to read article text and design ONE clear visual concept that:
1. Captures the CORE idea or analogy from the text
2. Uses a relatable real-world metaphor (computers, buildings, nature, everyday objects)
3. Can be expressed as a comparison, flow diagram, or analogy side-by-side
4. Works as a hand-drawn sketch with labeled parts and a bottom caption

Output ONLY a JSON object (no markdown fences) with these fields:
{
  "concept_title": "short title for the visual",
  "visual_metaphor": "the core metaphor or analogy being illustrated",
  "layout": "side-by-side comparison | flow diagram | single scene with annotations | hierarchy",
  "elements": ["list", "of", "visual", "elements", "to", "draw"],
  "labels": ["key", "labels", "and", "annotations"],
  "caption": "The witty bottom caption summarizing the insight",
  "dalle_prompt": "Complete, detailed DALL-E 3 prompt to generate this image"
}

The dalle_prompt MUST:
- Describe exact visual elements and their spatial positions
- Specify the hand-drawn sketch aesthetic on cream paper
- List all text labels to include in the image
- Include the bottom caption verbatim
- Be under 4000 characters
"""


# ── Core logic ─────────────────────────────────────────────────────────────────


def get_clients():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set")
    return anthropic.Anthropic(api_key=anthropic_key), OpenAI(api_key=openai_key)


def analyze_article(text: str, claude: anthropic.Anthropic) -> dict:
    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=CLAUDE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Design a hand-drawn educational illustration for this article:\n\n{text[:8000]}",
            }
        ],
    )
    raw = next(b.text for b in response.content if b.type == "text")
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Claude did not return valid JSON:\n{raw}")
    return json.loads(json_match.group())


def build_prompt(concept: dict) -> str:
    base = concept.get("dalle_prompt", "")
    if not base:
        base = (
            f"Educational hand-drawn diagram: {concept['concept_title']}. "
            f"Visual metaphor: {concept['visual_metaphor']}. "
            f"Layout: {concept['layout']}. "
            f"Elements: {', '.join(concept['elements'])}. "
            f"Labels: {', '.join(concept['labels'])}. "
            f'Bottom caption: \"{concept["caption"]}\".'
        )
    return f"{base}\n\nArt style: {STYLE_GUIDE}"


def generate_image(prompt: str, oai: OpenAI) -> str:
    resp = oai.images.generate(
        model="dall-e-3", prompt=prompt, size="1792x1024", quality="hd", n=1
    )
    return resp.data[0].url


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len]


def download_image(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No article text provided"}), 400

    try:
        claude, oai = get_clients()

        # Step 1 — Claude designs the concept
        concept = analyze_article(text, claude)

        # Step 2 — Build DALL-E prompt
        prompt = build_prompt(concept)

        # Step 3 — Generate image
        url = generate_image(prompt, oai)

        # Step 4 — Save locally
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = slugify(concept.get("concept_title", "image"))
        filename = f"{ts}_{slug}.png"
        download_image(url, OUTPUT_DIR / filename)

        return jsonify(
            {
                "concept": concept,
                "image_url": url,
                "local_file": f"/images/{filename}",
            }
        )

    except EnvironmentError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Generation failed: {e}"}), 500


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/gallery")
def gallery():
    images = sorted(OUTPUT_DIR.glob("*.png"), reverse=True)
    files = [f"/images/{img.name}" for img in images]
    return jsonify(files)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
