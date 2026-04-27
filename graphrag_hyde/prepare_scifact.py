"""
prepare_scifact.py
------------------
Loads the SciFact corpus from HuggingFace and writes each document as a
separate .txt file under graphrag_quickstart/input/.

GraphRAG indexes every file it finds in input/, so naming each file
<doc_id>.txt lets us recover the SciFact corpus ID from the indexed
document title after indexing.

Also saves input/doc_id_mapping.json  (doc_id -> full text) for reference.

Run once before `graphrag index`:
    cd graphrag_quickstart
    python prepare_scifact.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow importing from the sibling hydeOnSciFact package
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "hydeOnSciFact"))

from load_data import load_scifact_data  # noqa: E402

INPUT_DIR = Path(__file__).parent / "input"
MAPPING_PATH = Path(__file__).parent / "doc_id_mapping.json"


def main() -> None:
    INPUT_DIR.mkdir(exist_ok=True)

    print("Loading SciFact corpus …")
    corpus, queries, qrels, source = load_scifact_data(split="test")
    print(f"  Source : {source}")
    print(f"  Corpus : {len(corpus):,} documents")
    print(f"  Queries: {len(queries):,}  |  Qrels: {len(qrels):,}")

    # Write one file per corpus document
    written = 0
    for doc_id, text in corpus.items():
        out_path = INPUT_DIR / f"{doc_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        written += 1

    print(f"Wrote {written:,} files to {INPUT_DIR}/")

    # Save mapping for quick reference (not needed by GraphRAG itself)
    with MAPPING_PATH.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print(f"Saved doc_id_mapping.json  ({MAPPING_PATH})")

    print("\nNext step:")
    print("  graphrag index   # from inside graphrag_quickstart/")


if __name__ == "__main__":
    main()
