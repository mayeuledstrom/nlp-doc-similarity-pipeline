# NLP Document Similarity Pipeline

Small, production‑ready example of a **document similarity** pipeline in Python.

- **Use case:** quickly find similar documents (TXT/PDF) in a directory.
- **Tech:** Python, scikit‑learn (TF‑IDF), cosine similarity.
- **Why this repo:** demonstrates clean code, CLI entry point, tests, and reproducible env.

> Built to mirror a real internship task: *“pipeline NLP d'analyse de similarité sémantique de documents”* (Python, vectorisation, similarité cosinus).

## Features
- Load and normalize text from a folder (`.txt` by default).
- Vectorize with **TF‑IDF** and compute **cosine similarity**.
- Query a document (by index or file path) and print the **top‑k most similar**.
- Simple **CLI**: `python -m src.similarity_pipeline --folder data/sample_docs --k 3`

## Quickstart

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run on sample docs
python -m src.similarity_pipeline --folder data/sample_docs --k 3
```

Example output:
```
Loaded 4 documents from data/sample_docs
[Query] 0 - contract_clause.txt
Top 3 similar documents:
  1) nda_template.txt                 similarity=0.47
  2) energy_report.txt                similarity=0.11
  3) meeting_minutes.txt              similarity=0.06
```

## API / CLI

```bash
python -m src.similarity_pipeline --folder <path> --k 5 --query_index 0
# or
python -m src.similarity_pipeline --folder <path> --k 5 --query_file <path/file.txt>
```

Options:
- `--folder`: directory containing `.txt` files
- `--k`: number of neighbors to display
- `--query_index`: pick the document index as query
- `--query_file`: path to a specific file to use as query

## Project structure
```
nlp-doc-similarity-pipeline/
├── data/
│   └── sample_docs/
│       ├── contract_clause.txt
│       ├── nda_template.txt
│       ├── energy_report.txt
│       └── meeting_minutes.txt
├── src/
│   ├── __init__.py
│   ├── io_utils.py
│   └── similarity_pipeline.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Notes
- This repo keeps dependencies **lightweight** on purpose.
- For **semantic embeddings** (e.g. `sentence-transformers`), see the commented section in `similarity_pipeline.py`.

## License
MIT
