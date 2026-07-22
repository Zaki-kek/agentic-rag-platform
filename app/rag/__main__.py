"""Module entry point: ``python -m app.rag`` runs the offline ingest CLI."""

from __future__ import annotations

import sys

from app.rag.ingest_cli import main

if __name__ == "__main__":
    sys.exit(main())
