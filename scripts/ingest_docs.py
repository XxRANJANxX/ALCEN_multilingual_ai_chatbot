"""Ingest every file under DOCS_DIR:  python -m scripts.ingest_docs [dir]"""
import os
import sys

from app.core.container import get_container
from app.rag.ingest import IngestError, ingest_bytes

c = get_container()
root = sys.argv[1] if len(sys.argv) > 1 else c.settings.docs_dir
for dirpath, _, files in os.walk(root):
    for name in sorted(files):
        with open(os.path.join(dirpath, name), "rb") as f:
            try:
                print(ingest_bytes(c.store, name, f.read(), c.settings))
            except IngestError as e:
                print({"source": name, "error": str(e)})
