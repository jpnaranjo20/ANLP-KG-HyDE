from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

from datasets import load_dataset


Corpus = Dict[str, str]
Queries = Dict[str, str]
Qrels = Dict[str, Set[str]]


def _build_doc_text(title: str, abstract) -> str:
    if isinstance(abstract, list):
        abstract_text = " ".join([x for x in abstract if isinstance(x, str)])
    else:
        abstract_text = abstract if isinstance(abstract, str) else ""
    title = title if isinstance(title, str) else ""
    return f"Title: {title}\nAbstract: {abstract_text}".strip()


def _load_one(repo: str, config: str | None, split: str):
    if config is None:
        return load_dataset(repo, split=split)
    return load_dataset(repo, config, split=split)


def _try_load_dataset_component(repo: str, configs: list[str | None], splits: list[str]):
    last_error = None
    for cfg in configs:
        for sp in splits:
            try:
                ds = _load_one(repo, cfg, sp)
                return ds, cfg, sp
            except Exception as e:  # pragma: no cover
                last_error = e
    raise RuntimeError(f"Failed loading component for {repo}. Last error: {last_error}")


def _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds) -> Tuple[Corpus, Queries, Qrels]:
    corpus: Corpus = {}
    for row in corpus_ds:
        doc_id = str(row.get("_id", row.get("doc_id", row.get("id"))))
        if doc_id == "None":
            continue
        text = row.get("text", row.get("contents", ""))
        title = row.get("title", "")
        if title and text:
            corpus[doc_id] = f"Title: {title}\nPassage: {text}"
        elif text:
            corpus[doc_id] = text
        else:
            corpus[doc_id] = _build_doc_text(title, row.get("abstract", ""))

    queries: Queries = {}
    for row in queries_ds:
        qid = str(row.get("_id", row.get("query_id", row.get("id"))))
        query_text = row.get("text", row.get("query", row.get("claim", "")))
        if qid != "None" and isinstance(query_text, str) and query_text.strip():
            queries[qid] = query_text

    qrels: Qrels = defaultdict(set)
    for row in qrels_ds:
        qid = str(row.get("query-id", row.get("query_id", row.get("qid", row.get("id")))))
        did = str(row.get("corpus-id", row.get("doc_id", row.get("did"))))
        score = row.get("score", row.get("label", 1))

        if qid == "None" or did == "None":
            continue
        if isinstance(score, (int, float)) and score <= 0:
            continue
        qrels[qid].add(did)

    filtered_queries: Queries = {}
    filtered_qrels: Qrels = {}
    for qid, query in queries.items():
        gold = {did for did in qrels.get(qid, set()) if did in corpus}
        if gold:
            filtered_queries[qid] = query
            filtered_qrels[qid] = gold

    return corpus, filtered_queries, filtered_qrels


def _try_load_beir_scifact(split: str) -> Tuple[Corpus, Queries, Qrels, str]:
    # Prefer canonical BEIR repo names first.
    repos = ["BeIR/scifact", "beir/scifact", "mteb/scifact"]

    errors: list[str] = []
    for repo in repos:
        try:
            corpus_ds, corpus_cfg, corpus_split = _try_load_dataset_component(
                repo,
                configs=["corpus", None],
                splits=["corpus", "train", "test"],
            )
            queries_ds, queries_cfg, queries_split = _try_load_dataset_component(
                repo,
                configs=["queries", None],
                splits=[split, "queries", "test", "validation", "train"],
            )
            qrels_ds, qrels_cfg, qrels_split = _try_load_dataset_component(
                repo,
                configs=["qrels", None],
                splits=[split, "test", "validation", "train"],
            )

            corpus, queries, qrels = _parse_beir_triplet(corpus_ds, queries_ds, qrels_ds)
            if corpus and queries and qrels:
                source = (
                    f"{repo}"
                    f" (corpus:{corpus_cfg or 'default'}/{corpus_split},"
                    f" queries:{queries_cfg or 'default'}/{queries_split},"
                    f" qrels:{qrels_cfg or 'default'}/{qrels_split})"
                )
                return corpus, queries, qrels, source
            errors.append(f"{repo}: loaded but parsed empty corpus/queries/qrels")
        except Exception as e:
            errors.append(f"{repo}: {type(e).__name__}: {e}")

    raise RuntimeError(
        "Unable to load BEIR-style SciFact from Hugging Face Hub. "
        "Tried repos: BeIR/scifact, beir/scifact, mteb/scifact. "
        "Also note: old dataset-script `scifact` is unsupported in new `datasets` versions. "
        f"Details: {' | '.join(errors)}"
    )


def load_scifact_data(
    split: str = "test",
    prefer_beir: bool = True,
    max_queries: int | None = None,
) -> Tuple[Corpus, Queries, Qrels, str]:
    """
    Returns: corpus, queries, qrels, source_name
    """
    if not prefer_beir:
        raise RuntimeError(
            "Only BEIR-style loading is supported in this project now, "
            "because legacy `scifact` dataset scripts are deprecated by `datasets`."
        )

    corpus, queries, qrels, source = _try_load_beir_scifact(split=split)

    if max_queries is not None:
        keep_ids = list(queries.keys())[:max_queries]
        queries = {qid: queries[qid] for qid in keep_ids}
        qrels = {qid: qrels[qid] for qid in keep_ids}

    if not corpus or not queries or not qrels:
        raise RuntimeError("Loaded empty corpus/queries/qrels. Please check dataset access and split.")

    return corpus, queries, qrels, source
