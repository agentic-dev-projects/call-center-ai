# db/chroma_client.py

import chromadb

# chromadb.Client() is ephemeral (in-memory only) in v0.4+.
# PersistentClient writes to disk and survives process restarts.
client = chromadb.PersistentClient(path="./chroma_cache")