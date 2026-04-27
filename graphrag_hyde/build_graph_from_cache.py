"""
build_graph_from_cache.py
-------------------------
Builds GraphRAG output parquets directly from the extract_graph cache,
skipping the summarize_descriptions LLM step entirely.

Steps:
  1. Run GraphRAG's extractor on all text units (all cache hits = no API calls
     for the ~5259 cached chunks; ~296 uncached chunks make real API calls)
  2. Deduplicate entities and relationships without LLM summarization
  3. Compute entity degrees
  4. Run Leiden community detection (no LLM)
  5. Write all required parquet files to output/
  6. Embed entity descriptions and write to LanceDB

Run from inside graphrag_quickstart/:
    cd graphrag_quickstart
    python build_graph_from_cache.py
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent   # graphrag_quickstart/
REPO_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Helpers

def _dedup_list(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


# Main pipeline

async def main() -> None:
    from graphrag.cache.cache_key_creator import cache_key_creator
    from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
    from graphrag.config.enums import AsyncType
    from graphrag.config.load_config import load_config
    from graphrag.index.operations.cluster_graph import cluster_graph
    from graphrag.index.operations.extract_graph.extract_graph import (
        extract_graph as extractor,
    )
    from graphrag_cache.json_cache import JsonCache
    from graphrag_llm.completion import create_completion
    from graphrag_storage import create_storage
    from graphrag_storage.storage_config import StorageConfig

    # 1. Load config
    logger.info("Loading GraphRAG config from %s", SCRIPT_DIR)
    config = load_config(root_dir=SCRIPT_DIR)

    # 2. Create extraction model pointing at existing cache
    logger.info("Setting up extraction model (cache: cache/extract_graph/)...")
    model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    cache_storage = create_storage(
        StorageConfig(
            type="file",
            base_dir=str(SCRIPT_DIR / "cache" / "extract_graph"),
        )
    )
    extraction_model = create_completion(
        model_config,
        cache=JsonCache(storage=cache_storage),
        cache_key_creator=cache_key_creator,
    )

    # 3. Load text units
    logger.info("Loading text_units.parquet...")
    text_units = pd.read_parquet(OUTPUT_DIR / "text_units.parquet")
    logger.info("  %d text units", len(text_units))

    # 4. Run extractor (cache hits for already-processed chunks)
    logger.info("Running extractor (cache hits are instant)...")
    extraction_prompt = config.extract_graph.resolved_prompts().extraction_prompt
    entity_types = config.extract_graph.entity_types

    raw_entities, raw_relationships = await extractor(
        text_units=text_units,
        callbacks=NoopWorkflowCallbacks(),
        text_column="text",
        id_column="id",
        model=extraction_model,
        prompt=extraction_prompt,
        entity_types=entity_types,
        max_gleanings=config.extract_graph.max_gleanings,
        num_threads=8,
        async_type=AsyncType.Threaded,
    )

    logger.info(
        "  Raw: %d entity rows, %d relationship rows",
        len(raw_entities), len(raw_relationships),
    )

    if raw_entities.empty:
        logger.error("No entities extracted — aborting.")
        sys.exit(1)

    # 5. Flatten entities (extractor already grouped by title)
    # raw_entities columns: title, type, description (list), text_unit_ids (list), frequency
    logger.info("Flattening entities...")

    entity_agg = raw_entities.copy()
    entity_agg["description"] = entity_agg["description"].apply(
        lambda x: " | ".join(x) if isinstance(x, list) else str(x)
    )
    entity_agg["text_unit_ids"] = entity_agg["text_unit_ids"].apply(
        lambda x: _dedup_list(x) if isinstance(x, list) else [x]
    )
    entity_agg["id"] = [str(uuid.uuid4()) for _ in range(len(entity_agg))]
    entity_agg["human_readable_id"] = range(len(entity_agg))
    logger.info("  %d unique entities", len(entity_agg))

    # 6. Flatten relationships (extractor already grouped by source/target)
    # raw_relationships columns: source, target, description (list), text_unit_ids (list), weight
    logger.info("Flattening relationships...")

    rel_agg = raw_relationships.copy()
    rel_agg["description"] = rel_agg["description"].apply(
        lambda x: " | ".join(x) if isinstance(x, list) else str(x)
    )
    rel_agg["text_unit_ids"] = rel_agg["text_unit_ids"].apply(
        lambda x: _dedup_list(x) if isinstance(x, list) else [x]
    )
    rel_agg["id"] = [str(uuid.uuid4()) for _ in range(len(rel_agg))]
    rel_agg["human_readable_id"] = range(len(rel_agg))
    logger.info("  %d unique relationships", len(rel_agg))

    # 7. Compute entity degrees
    logger.info("Computing entity degrees...")
    degree_map: dict[str, int] = defaultdict(int)
    seen_pairs: set[tuple[str, str]] = set()
    for _, row in rel_agg.iterrows():
        pair = tuple(sorted([row["source"], row["target"]]))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            degree_map[row["source"]] += 1
            degree_map[row["target"]] += 1

    entity_agg["degree"] = entity_agg["title"].map(lambda t: degree_map.get(t, 0))
    rel_agg["combined_degree"] = rel_agg.apply(
        lambda r: degree_map.get(r["source"], 0) + degree_map.get(r["target"], 0),
        axis=1,
    )

    # 8. Write entities.parquet and relationships.parquet
    logger.info("Writing entities.parquet...")
    entity_agg["description_embedding"] = None
    entity_agg.to_parquet(OUTPUT_DIR / "entities.parquet", index=False)

    logger.info("Writing relationships.parquet...")
    rel_agg.to_parquet(OUTPUT_DIR / "relationships.parquet", index=False)

    # 9. Community detection (Leiden, no LLM)
    logger.info("Running community detection...")
    clusters = cluster_graph(
        rel_agg,
        max_cluster_size=config.cluster_graph.max_cluster_size,
        use_lcc=getattr(config.cluster_graph, "use_lcc", True),
        seed=getattr(config.cluster_graph, "seed", None),
    )

    title_to_id = dict(zip(entity_agg["title"], entity_agg["id"]))

    communities_exploded = (
        pd.DataFrame(clusters, columns=pd.Index(["level", "community", "parent", "title"]))
        .explode("title")
    )
    communities_exploded["community"] = communities_exploded["community"].astype(int)

    entity_id_map = communities_exploded[["community", "title"]].copy()
    entity_id_map["entity_id"] = entity_id_map["title"].map(title_to_id)
    entity_ids_df = (
        entity_id_map.dropna(subset=["entity_id"])
        .groupby("community")
        .agg(entity_ids=("entity_id", list))
        .reset_index()
    )

    # Intra-community relationships per level
    level_results = []
    for level in communities_exploded["level"].unique():
        lvl = communities_exploded[communities_exploded["level"] == level]
        with_src = rel_agg.merge(lvl, left_on="source", right_on="title", how="inner")
        with_both = with_src.merge(lvl, left_on="target", right_on="title", how="inner")
        intra = with_both[with_both["community_x"] == with_both["community_y"]]
        if intra.empty:
            continue
        grouped = (
            intra.explode("text_unit_ids")
            .groupby(["community_x", "parent_x"])
            .agg(
                relationship_ids=("id", list),
                text_unit_ids=("text_unit_ids", list),
            )
            .reset_index()
            .rename(columns={"community_x": "community", "parent_x": "parent"})
        )
        grouped["level"] = level
        level_results.append(grouped)

    if level_results:
        all_grouped = pd.concat(level_results, ignore_index=True)
        all_grouped["relationship_ids"] = all_grouped["relationship_ids"].apply(
            lambda x: sorted(set(x))
        )
        all_grouped["text_unit_ids"] = all_grouped["text_unit_ids"].apply(
            lambda x: sorted(set(str(t) for t in x if t is not None))
        )
        final_communities = all_grouped.merge(entity_ids_df, on="community", how="inner")
    else:
        # fallback: single top-level community with all entities
        final_communities = pd.DataFrame([{
            "community": 0, "parent": -1, "level": 0,
            "relationship_ids": rel_agg["id"].tolist(),
            "text_unit_ids": [],
            "entity_ids": entity_agg["id"].tolist(),
        }])

    final_communities["id"] = [str(uuid.uuid4()) for _ in range(len(final_communities))]
    final_communities["human_readable_id"] = final_communities["community"]
    final_communities["title"] = "Community " + final_communities["community"].astype(str)
    final_communities["parent"] = final_communities["parent"].astype(int)

    parent_grouped = (
        final_communities.groupby("parent")
        .agg(children=("community", lambda x: x.tolist()))
        .reset_index()
    )
    final_communities = final_communities.merge(
        parent_grouped, left_on="community", right_on="parent", how="left"
    )
    final_communities["children"] = final_communities["children"].apply(
        lambda x: x if isinstance(x, list) else []
    )
    final_communities["period"] = datetime.now(timezone.utc).date().isoformat()
    final_communities["size"] = final_communities["entity_ids"].apply(len)

    logger.info("Writing communities.parquet (%d communities)...", len(final_communities))
    final_communities.to_parquet(OUTPUT_DIR / "communities.parquet", index=False)

    # 10. Write empty community_reports.parquet
    logger.info("Writing empty community_reports.parquet...")
    empty_reports = pd.DataFrame(columns=[
        "id", "human_readable_id", "community", "level", "parent",
        "title", "summary", "full_content", "rank",
    ])
    empty_reports.to_parquet(OUTPUT_DIR / "community_reports.parquet", index=False)

    # 11. Embed entity descriptions → LanceDB
    logger.info("Generating entity description embeddings...")
    await _embed_entities(config, entity_agg)

    logger.info("Done! Output files written to %s", OUTPUT_DIR)
    logger.info("You can now run the retrieval notebook.")


async def _embed_entities(config, entity_agg: pd.DataFrame) -> None:
    from graphrag.cache.cache_key_creator import cache_key_creator
    from graphrag.config.embeddings import entity_description_embedding
    from graphrag_cache.json_cache import JsonCache
    from graphrag_llm.embedding import create_embedding
    from graphrag_storage import create_storage
    from graphrag_storage.storage_config import StorageConfig
    from graphrag_vectors import VectorStoreDocument, create_vector_store

    embed_model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    embed_cache_storage = create_storage(
        StorageConfig(
            type="file",
            base_dir=str(SCRIPT_DIR / "cache" / "embed_text"),
        )
    )
    embed_model = create_embedding(
        embed_model_config,
        cache=JsonCache(storage=embed_cache_storage),
        cache_key_creator=cache_key_creator,
    )

    vector_store = create_vector_store(
        config.vector_store,
        config.vector_store.index_schema[entity_description_embedding],
    )
    vector_store.connect()
    vector_store.create_index()  # creates the LanceDB table and sets document_collection

    texts = [
        f"{row['title']}:{row['description']}"
        for _, row in entity_agg.iterrows()
    ]

    BATCH = 100
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        response = await embed_model.embedding_async(input=batch)
        all_vectors.extend(response.embeddings)
        logger.info(
            "  Embedded %d / %d entities", min(i + BATCH, len(texts)), len(texts)
        )

    documents = [
        VectorStoreDocument(
            id=str(row["id"]),
            vector=all_vectors[idx],
            data={"title": row["title"]},
        )
        for idx, (_, row) in enumerate(entity_agg.iterrows())
    ]
    vector_store.load_documents(documents)
    logger.info("  Wrote %d vectors to LanceDB.", len(documents))


if __name__ == "__main__":
    asyncio.run(main())
