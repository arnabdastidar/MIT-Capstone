"""
Article Image Generator
-----------------------
Generates hand-drawn style educational illustrations for articles and concepts
using Claude (for visual design) + DALL-E 3 (for image generation).

Usage:
    python article_image_generator.py --text "your article text here"
    python article_image_generator.py --file article.txt
    python article_image_generator.py --file article.txt --count 3
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Style constants ────────────────────────────────────────────────────────────
STYLE_GUIDE = """
hand-drawn educational diagram, black ink on cream/beige paper background,
sketch style, simple cartoon characters with minimal detail, clean outlines,
annotated with clear bold labels and arrows, thought bubbles and speech bubbles,
comparison/analogy layout, friendly and approachable visual metaphor,
no color fills except light gray for shading, bottom caption in italic font,
similar to "wait but why" or "sketchplanations" illustration style
"""

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

# ── Core functions ─────────────────────────────────────────────────────────────

def analyze_article(text: str, client: anthropic.Anthropic) -> list[dict]:
    """Use Claude to design 5 distinct visual concepts from article text."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8192,
        thinking={"type": "adaptive"},
        system=CLAUDE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Design 5 hand-drawn educational illustrations for this article:\n\n{text[:8000]}"
            }
        ]
    )

    raw = next(
        block.text for block in response.content if block.type == "text"
    )

    import json
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Claude did not return valid JSON array:\n{raw}")
    concepts = json.loads(json_match.group())
    if not isinstance(concepts, list) or len(concepts) == 0:
        raise ValueError("Claude returned an empty or invalid concept list")
    return concepts[:5]


def build_dalle_prompt(concept: dict) -> str:
    """Combine concept details with the style guide into a DALL-E 3 prompt."""
    base = concept.get("dalle_prompt", "")
    if not base:
        # Fallback: construct from parts
        base = (
            f"Educational hand-drawn diagram: {concept['concept_title']}. "
            f"Visual metaphor: {concept['visual_metaphor']}. "
            f"Layout: {concept['layout']}. "
            f"Elements: {', '.join(concept['elements'])}. "
            f"Labels: {', '.join(concept['labels'])}. "
            f'Bottom caption text: "{concept["caption"]}".'
        )
    return f"{base}\n\nArt style: {STYLE_GUIDE.strip()}"


def generate_image(prompt: str, oai_client: OpenAI, size: str = "1792x1024") -> str:
    """Call DALL-E 3 and return the image URL."""
    response = oai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality="hd",
        n=1,
    )
    return response.data[0].url


def download_image(url: str, output_path: Path) -> Path:
    """Download image from URL and save to disk."""
    import urllib.request
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def slugify(text: str, max_len: int = 40) -> str:
    """Convert text to a safe filename slug."""
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[\s_-]+', '_', slug).strip('_')
    return slug[:max_len]


# ── Main pipeline ──────────────────────────────────────────────────────────────

def generate_article_images(
    article_text: str,
    output_dir: str = "generated_images",
    verbose: bool = True,
) -> list[dict]:
    """
    Full pipeline: article text → 5 Claude concepts → 5 DALL-E 3 images → saved files.

    Returns a list of dicts, each with keys: concept, prompt, url, file_path
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment or .env file")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment or .env file")

    claude = anthropic.Anthropic(api_key=anthropic_key)
    oai = OpenAI(api_key=openai_key)

    # Step 1: Claude designs 5 visual concepts
    if verbose:
        print("🧠 Analyzing article and designing 5 visual concepts...")
    concepts = analyze_article(article_text, claude)

    if verbose:
        for i, c in enumerate(concepts, 1):
            print(f"   {i}. {c['concept_title']} — {c['visual_metaphor']}")

    # Step 2-4: Generate each image
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, concept in enumerate(concepts, 1):
        if verbose:
            print(f"\n🎨 Generating image {i}/{len(concepts)}: {concept['concept_title']}...")

        prompt = build_dalle_prompt(concept)
        url = generate_image(prompt, oai)

        slug = slugify(concept["concept_title"])
        filename = f"{timestamp}_{i}_{slug}.png"
        output_path = Path(output_dir) / filename

        if verbose:
            print(f"   💾 Saved: {output_path}")
        download_image(url, output_path)

        results.append({
            "concept": concept,
            "prompt": prompt,
            "url": url,
            "file_path": str(output_path),
        })

    if verbose:
        print(f"\n✅ Done! Generated {len(results)} images.")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate hand-drawn style illustrations for articles using Claude + DALL-E 3"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", "-t", help="Article text directly as a string")
    source.add_argument("--file", "-f", help="Path to a text file containing the article")

    parser.add_argument(
        "--output", "-o",
        default="generated_images",
        help="Output directory for saved images (default: generated_images)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    # Read article text
    if args.text:
        article_text = args.text
    else:
        path = Path(args.file)
        if not path.exists():
            print(f"Error: file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        article_text = path.read_text(encoding="utf-8")

    verbose = not args.quiet

    # Generate 5 images
    results = generate_article_images(article_text, args.output, verbose)
    print(f"\nGenerated {len(results)} images:")
    for r in results:
        print(f"  {r['file_path']}")


if __name__ == "__main__":
    main()
