import os
from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"

# Dataset
# Prefer BEIR-formatted SciFact first, fallback to raw scifact if unavailable.
PREFER_BEIR = True
DATASET_SPLIT = "test"
MAX_QUERIES = None

# Models
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "gpt-4o-mini"
HYDE_FALLBACK_MODELS = []
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HYDE_PROVIDER = "openai"
HF_INFERENCE_PROVIDER = "hf-inference"
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# HyDE generation
HYDE_PROMPT_TEMPLATE = (
    "Given a scientific claim or question, write a short scientific passage that would answer it.\n"
    "The passage should be factual in style, concise, and relevant to the claim.\n\n"
    "Claim: {query}\n\n"
    "Passage:"
)
HYDE_MAX_NEW_TOKENS = 128
HYDE_TEMPERATURE = 0.3
HYDE_DO_SAMPLE = False

# Retrieval / evaluation
TOP_KS = [1, 10, 100]
EMBED_BATCH_SIZE = 64

# Runtime
SEED = 42
