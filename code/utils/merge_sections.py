# merge_sections.py
import os

SECTIONS_DIR = "sections"
OUTPUT_PATH = "paper.md"

SECTION_ORDER = [
    "01_intro.md",
    "02_theory.md",
    "03_experiments.md",
    "04_results.md",
    "05_discussion.md",
    "06_conclusion.md",
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for fname in SECTION_ORDER:
        path = os.path.join(SECTIONS_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        out.write(content)
        out.write("\n\n---\n\n")

print(f"Merged into {OUTPUT_PATH}")