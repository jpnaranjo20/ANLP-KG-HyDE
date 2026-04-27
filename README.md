# ANLP-HW34 — HyDE Reproduction & KG-Grounded HyDE

Repo for **CMU 11-711 Advanced NLP (Spring 2026), Assignment 3 & 4**.

We reproduce the [HyDE paper](https://arxiv.org/abs/2212.10496) (Hypothetical Document Embeddings for zero-shot dense retrieval) and extend it with a knowledge-graph-grounded variant on SciFact. The repo contains three retrieval tracks side-by-side.

Assignment instructions: <https://cmu-l3.github.io/anlp-spring2026/assignments/assignment3&4>

Project proposal: [`project_proposal.pdf`](project_proposal.pdf)

Final write-up: TODO

## Team

- Wenguang Dong — wenguand@andrew.cmu.edu
- Yichen Ji — yichenj@andrew.cmu.edu
- Juan Pablo Naranjo — jpnaranj@andrew.cmu.edu

## Repository layout

| Path | What it is |
|---|---|
| [`hyde/`](hyde/) | Faithful reproduction of the original paper on **TREC DL19**, with the generator switched from GPT-3 to GPT-4o-mini. Uses pyserini + Contriever. See [`hyde/README.md`](hyde/README.md). |
| [`hydeOnSciFact/`](hydeOnSciFact/) | HyDE applied to **SciFact** with `BAAI/bge-base-en-v1.5` embeddings and GPT-4o-mini. See [`hydeOnSciFact/README.md`](hydeOnSciFact/README.md). |
| [`graphrag_hyde/`](graphrag_hyde/) | **KG-Grounded HyDE (headline extension).** Builds a Microsoft GraphRAG knowledge graph + community summaries over SciFact, then layers HyDE on top of graph retrieval. |
| [`all_results/`](all_results/) | Committed metrics and run files for DL19 (`bm25`, `contriever`, `hyde`) and SciFact (`bge_baseline`, `bge_hyde`, `bm25`, `contriever_baseline`, `contriever_hyde`). |
| [`result_error_analysis.ipynb`](result_error_analysis.ipynb) | Query-level error analysis on DL19: see [Error analysis](#error-analysis) below. |
| [`project_proposal.pdf`](project_proposal.pdf) | Project proposal report. |
| [`report.pdf`](report.pdf) | Final report. |

## Quick start

Pick the track you want to run. Each subdirectory is independent.

### 1. HyDE on TREC DL19 (paper reproduction)

```bash
cd hyde
pip install -e .
# Download the prebuilt Contriever FAISS index over MS MARCO passages:
wget https://www.dropbox.com/s/dytqaqngaupp884/contriever_msmarco_index.tar.gz
tar -xvf contriever_msmarco_index.tar.gz
export OPENAI=<your-openai-key>
# Then open:
#   hyde-dl19.ipynb   (full DL19 run)
#   hyde-demo.ipynb   (single-query walkthrough)
```

Also requires `pyserini` — see its [install guide](https://github.com/castorini/pyserini#-installation). Full instructions in [`hyde/README.md`](hyde/README.md).

### 2. HyDE on SciFact

```bash
cd hydeOnSciFact
pip install -r requirements.txt
export OPENAI_API_KEY=<your-openai-key>
export HF_TOKEN=<your-hf-token>
# Then open:
#   run_baseline.ipynb     (BGE dense baseline)
#   run_hyde.ipynb         (BGE + HyDE)
#   run_bm25.ipynb         (BM25 baseline)
#   run_contriever.ipynb   (Contriever baseline)
```

### 3. KG-Grounded HyDE on SciFact (extension)

```bash
cd graphrag_hyde
# Configure GRAPHRAG_API_KEY in .env (see settings.yaml).
# 1. Convert the SciFact corpus into GraphRAG's input format:
python prepare_scifact.py
# 2. Build the knowledge graph + community reports (expensive; cached):
python build_graph_from_cache.py
# 3. Then open the retrieval notebooks:
#   run_graphrag_retrieval.ipynb   (GraphRAG local search)
#   run_graph_hyde.ipynb           (HyDE + graph retrieval — best results)
```

GraphRAG uses `gpt-4o-mini` for completion and `text-embedding-3-large` for embeddings (configured in [`graphrag_hyde/settings.yaml`](graphrag_hyde/settings.yaml)).

## Headline results — SciFact (300 test queries, 5,183 corpus docs)

| System | Recall@10 | Recall@100 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|
| BGE dense (baseline) | 85.66 | 96.33 | 69.36 | 72.96 |
| BGE + HyDE | 88.24 | 96.67 | 69.01 | 73.38 |
| **BGE + KG-Grounded HyDE** | **89.51** | **97.33** | **72.21** | **75.88** |

Sources: [`all_results/scifact/bge_baseline/metrics.json`](all_results/scifact/bge_baseline/metrics.json), [`all_results/scifact/bge_hyde/metrics.json`](all_results/scifact/bge_hyde/metrics.json), [`graphrag_hyde/results/graph_hyde_metrics.json`](graphrag_hyde/results/graph_hyde_metrics.json).

## Error analysis

[`result_error_analysis.ipynb`](result_error_analysis.ipynb) computes per-query metrics across the DL19 runs in [`all_results/dl19/`](all_results/dl19/), compares BM25 / Contriever / HyDE on nDCG@10 and Recall@10, and identifies queries where HyDE underperforms its baseline. It exports:

- `all_results/dl19/error_analysis/comparison_per_query.jsonl`
- `all_results/dl19/error_analysis/failure_cases.jsonl`
