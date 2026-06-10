# This file centralizes settings so you don't hardcode 
# them everywhere. Open src/config.py and paste:

import os
from dotenv import load_dotenv

load_dotenv(override=True)

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY      = os.getenv("PINECONE_API_KEY")
PINECONE_DENSE_INDEX  = os.getenv("PINECONE_DENSE_INDEX", "worldcup-rag-dense")
PINECONE_SPARSE_INDEX = os.getenv("PINECONE_SPARSE_INDEX", "worldcup-rag-sparse")

EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM      = 384
LLM_MODEL      = "claude-haiku-4-5-20251001"

CHUNK_SIZE     = 800   # characters; ~160-200 tokens, within all-MiniLM-L6-v2's 256-token limit
CHUNK_OVERLAP  = 100   # 12.5% overlap keeps context across boundaries
TOP_K          = 5
ALPHA          = 0.7
SCORE_THRESHOLD = 0.45