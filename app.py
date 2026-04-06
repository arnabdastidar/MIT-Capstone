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

Your job is to read article text and design FIVE distinct visual concepts that each
illustrate a different angle, section, or key idea from the text. Each concept should:
1. Capture a DIFFERENT core idea, analogy, or section from the article
2. Use a relatable real-world metaphor (computers, buildings, nature, everyday objects)
3. Use a DIFFERENT layout style so the set feels varied and rich
4. Work as a hand-drawn sketch with labeled parts and a bottom caption

Output ONLY a JSON array (no markdown fences) with exactly 5 objects, each having these fields:
[
  {
    "concept_title": "short title for the visual",
    "visual_metaphor": "the core metaphor or analogy being illustrated",
    "layout": "side-by-side comparison | flow diagram | single scene with annotations | hierarchy | table/matrix",
    "elements": ["list", "of", "visual", "elements", "to", "draw"],
    "labels": ["key", "labels", "and", "annotations"],
    "caption": "The witty bottom caption summarizing the insight",
    "dalle_prompt": "Complete, detailed DALL-E 3 prompt to generate this image"
  }
]

IMPORTANT RULES:
- Each of the 5 concepts must cover a DIFFERENT aspect of the article
- Vary the layouts: use at least 3 different layout types across the 5
- Each dalle_prompt must be self-contained and under 4000 characters
- dalle_prompts must specify the hand-drawn sketch aesthetic on cream paper
- Include all text labels and the bottom caption in each dalle_prompt
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


def analyze_article(text: str, claude: anthropic.Anthropic) -> list[dict]:
    """Ask Claude to design 5 distinct visual concepts for the article."""
    response = claude.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=CLAUDE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Design 5 hand-drawn educational illustrations for this article:\n\n{text[:8000]}",
            }
        ],
    )
    raw = next(b.text for b in response.content if b.type == "text")
    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Claude did not return valid JSON array:\n{raw}")
    concepts = json.loads(json_match.group())
    if not isinstance(concepts, list) or len(concepts) == 0:
        raise ValueError("Claude returned an empty or invalid concept list")
    return concepts[:5]


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

        # Step 1 — Claude designs 5 visual concepts
        concepts = analyze_article(text, claude)

        # Step 2–4 — Generate each image
        results = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, concept in enumerate(concepts):
            prompt = build_prompt(concept)
            url = generate_image(prompt, oai)
            slug = slugify(concept.get("concept_title", "image"))
            filename = f"{ts}_{i+1}_{slug}.png"
            download_image(url, OUTPUT_DIR / filename)
            results.append(
                {
                    "concept": concept,
                    "image_url": url,
                    "local_file": f"/images/{filename}",
                }
            )

        return jsonify({"images": results})

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
