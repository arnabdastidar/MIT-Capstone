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

Your job is to read article text and design ONE clear visual concept that:
1. Captures the CORE idea or analogy from the text
2. Uses a relatable real-world metaphor (computers, buildings, nature, everyday objects)
3. Can be expressed as a comparison, flow diagram, or analogy side-by-side
4. Works as a hand-drawn sketch with labeled parts and a bottom caption

Output ONLY a JSON object with these fields:
{
  "concept_title": "short title for the visual",
  "visual_metaphor": "the core metaphor or analogy being illustrated",
  "layout": "side-by-side comparison | flow diagram | single scene with annotations | hierarchy",
  "elements": ["list", "of", "visual", "elements", "to", "draw"],
  "labels": ["key", "labels", "and", "annotations"],
  "caption": "The witty bottom caption summarizing the insight",
  "dalle_prompt": "The complete, detailed DALL-E 3 prompt to generate this image"
}

The dalle_prompt must be specific about:
- Exact visual elements and their positions
- The hand-drawn sketch aesthetic
- Labels and text to include
- The bottom caption text
- The overall scene composition
"""

# ── Core functions ─────────────────────────────────────────────────────────────

def analyze_article(text: str, client: anthropic.Anthropic) -> dict:
    """Use Claude to extract the key visual concept from article text."""
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=CLAUDE_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Design a hand-drawn educational illustration for this article:\n\n{text[:6000]}"
            }
        ]
    )

    # Extract text content (thinking blocks come first)
    raw = next(
        block.text for block in response.content if block.type == "text"
    )

    # Parse JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"Claude did not return valid JSON:\n{raw}")

    import json
    return json.loads(json_match.group())


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

def generate_article_image(
    article_text: str,
    output_dir: str = "generated_images",
    verbose: bool = True,
) -> dict:
    """
    Full pipeline: article text → Claude concept → DALL-E 3 image → saved file.

    Returns a dict with keys: concept, prompt, url, file_path
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not anthropic_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set in environment or .env file")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY not set in environment or .env file")

    claude = anthropic.Anthropic(api_key=anthropic_key)
    oai = OpenAI(api_key=openai_key)

    # Step 1: Claude designs the visual concept
    if verbose:
        print("🧠 Analyzing article and designing visual concept...")
    concept = analyze_article(article_text, claude)

    if verbose:
        print(f"   Concept: {concept['concept_title']}")
        print(f"   Metaphor: {concept['visual_metaphor']}")
        print(f"   Caption: \"{concept['caption']}\"")

    # Step 2: Build the DALL-E prompt
    prompt = build_dalle_prompt(concept)

    if verbose:
        print("\n🎨 Generating image with DALL-E 3...")

    # Step 3: Generate image
    url = generate_image(prompt, oai)

    # Step 4: Save to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(concept["concept_title"])
    filename = f"{timestamp}_{slug}.png"
    output_path = Path(output_dir) / filename

    if verbose:
        print(f"💾 Saving image to {output_path}...")
    download_image(url, output_path)

    if verbose:
        print(f"\n✅ Done! Image saved: {output_path}")

    return {
        "concept": concept,
        "prompt": prompt,
        "url": url,
        "file_path": str(output_path),
    }


def generate_multiple(
    article_text: str,
    count: int = 3,
    output_dir: str = "generated_images",
    verbose: bool = True,
) -> list[dict]:
    """
    Generate multiple image variations for the same article.
    Claude re-analyzes each time to get different angles on the concept.
    """
    results = []
    for i in range(count):
        if verbose:
            print(f"\n{'='*50}")
            print(f"  Generating image {i+1}/{count}")
            print(f"{'='*50}")
        result = generate_article_image(article_text, output_dir, verbose)
        results.append(result)
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
        "--count", "-n",
        type=int,
        default=1,
        help="Number of image variations to generate (default: 1)"
    )
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

    # Generate
    if args.count == 1:
        result = generate_article_image(article_text, args.output, verbose)
        print(f"\nSaved: {result['file_path']}")
    else:
        results = generate_multiple(article_text, args.count, args.output, verbose)
        print(f"\nGenerated {len(results)} images:")
        for r in results:
            print(f"  {r['file_path']}")


if __name__ == "__main__":
    main()
