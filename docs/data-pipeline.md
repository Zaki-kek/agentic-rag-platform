# Data pipeline (ingestion)

How an uploaded document becomes searchable. The pipeline lives in
`app/rag/` and is orchestrated by `RagPipeline.ingest` (`app/rag/pipeline.py`),
driven from the `POST /documents` route.

```
upload -> extract -> chunk -> embed -> store        (ingest)
query  ------------> embed -> search                (retrieve)
```

## Stages

### 1. Extract (`app/rag/extract.py`)

Dispatches by file extension into plain text:

- `.pdf` -> PyMuPDF (`fitz`), page text joined by newlines;
- `.docx` -> python-docx, paragraph text joined by newlines;
- `.txt` / `.md` -> UTF-8 decode.

Heavy parsers (PyMuPDF, python-docx) are lazy-imported so the offline path and
plain-text ingestion do not pay for them.

### 2. Chunk (`app/rag/chunk.py`)

`chunk_text` splits on whitespace (word-boundary aware, so tokens are never cut
mid-word) into chunks of roughly `CHUNK_SIZE` characters, repeating
`CHUNK_OVERLAP` characters of trailing context between adjacent chunks to preserve
continuity across boundaries.

### 3. Embed (`app/rag/embed.py`)

Each chunk is turned into a vector by the configured `EMBEDDER`:

- `hash` - deterministic, offline, no network (default);
- `openai` - real embeddings via the OpenAI API.

### 4. Store (`app/rag/store.py`)

Vectors plus their source filename go into the configured `VECTOR_STORE`:

- `memory` - process-local, lost on restart (default);
- `pgvector` - Postgres with an HNSW cosine index for indexed ANN search.

`ingest` returns the number of chunks stored.

## Retrieval

`RagPipeline.retrieve` embeds the query with the **same** embedder and asks the
store for the top-`TOP_K` nearest chunks (cosine). Those chunks are what the answer
cites - retrieval and ingestion must share an embedder or vectors are not
comparable.

## Edge cases

- **Unsupported file type** - any extension outside `{.pdf, .docx, .txt, .md}`
  raises `UnsupportedFileTypeError`, surfaced as HTTP `415`
  (`unsupported_media_type`) with the uniform error body.
- **Empty / whitespace-only document, or a file that extracts to no text** -
  `chunk_text` returns an empty list; `ingest` logs a warning and returns `0`
  chunks instead of raising. The upload succeeds with `chunks: 0` and nothing is
  stored - retrieval simply won't surface it.
- **Corrupt PDF / DOCX** - a malformed file makes the underlying parser
  (PyMuPDF / python-docx) raise; the error propagates to the caller rather than
  storing partial garbage. The pipeline does not silently swallow parser failures.
- **Non-UTF-8 text / mixed encodings in `.txt` / `.md`** - decoded with
  `errors="replace"`, so a bad byte becomes the replacement character instead of
  failing the whole ingest.
- **Missing filename** - the route falls back to the name `"upload"` so extension
  dispatch still runs (and an extensionless `"upload"` hits the unsupported-type
  path).
- **Very large documents** - the whole file is read into memory and chunked in
  one pass; there is no streaming ingest yet. Size it against the host's memory,
  or split large corpora into multiple uploads.

## Configuration

`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `EMBEDDER`, `VECTOR_STORE` - see
[.env.example](../.env.example). Their cost implications are covered in the
[Cost controls](../README.md#cost-controls) section of the README.
