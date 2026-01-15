from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .io_utils import load_txt_folder

@dataclass
class Corpus:
    paths: List[str]
    texts: List[str]
    tfidf: Optional[np.ndarray] = None
    vectorizer: Optional[TfidfVectorizer] = None

def build_corpus(folder: str) -> Corpus:
    pairs = load_txt_folder(folder)
    if not pairs:
        raise SystemExit(f"No .txt files found in folder: {folder}")
    paths, texts = zip(*pairs)
    return Corpus(paths=list(paths), texts=list(texts))

def fit_tfidf(corpus: Corpus, max_features: int = 20000, ngram_range=(1,2)) -> Corpus:
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", max_features=max_features, ngram_range=ngram_range)
    X = vectorizer.fit_transform(corpus.texts)
    corpus.vectorizer = vectorizer
    corpus.tfidf = X
    return corpus

def most_similar(corpus: Corpus, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
    sims = cosine_similarity(query_vec, corpus.tfidf).ravel()
    # argsort descending, skip self if needed by filtering later
    idx = np.argsort(-sims)
    results = [(int(i), float(sims[i])) for i in idx[:k]]
    return results

def query_by_index(corpus: Corpus, index: int, k: int = 5) -> List[Tuple[int, float]]:
    if index < 0 or index >= len(corpus.paths):
        raise IndexError("query index out of range")
    qv = corpus.tfidf[index]
    results = most_similar(corpus, qv, k+1)  # +1 to include self, will filter
    results = [(i, s) for (i, s) in results if i != index][:k]
    return results

def query_by_text(corpus: Corpus, text: str, k: int = 5) -> List[Tuple[int, float]]:
    qv = corpus.vectorizer.transform([text])
    results = most_similar(corpus, qv, k)
    return results

def main():
    ap = argparse.ArgumentParser(description="NLP Document Similarity Pipeline (TF-IDF + Cosine)")
    ap.add_argument("--folder", required=True, help="Folder with .txt files")
    ap.add_argument("--k", type=int, default=5, help="Top-k neighbors to display")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--query_index", type=int, help="Use this document index as the query")
    group.add_argument("--query_file", type=str, help="Use this file's text as the query")
    args = ap.parse_args()

    corpus = build_corpus(args.folder)
    corpus = fit_tfidf(corpus)
    print(f"Loaded {len(corpus.paths)} documents from {args.folder}")
    for i, p in enumerate(corpus.paths):
        print(f"[{i}] {p.split('/')[-1]}")

    if args.query_index is not None:
        res = query_by_index(corpus, args.query_index, k=args.k)
        print(f"\n[Query] {args.query_index} - {corpus.paths[args.query_index].split('/')[-1]}")
    elif args.query_file is not None:
        from pathlib import Path
        qtext = Path(args.query_file).read_text(encoding="utf-8", errors="ignore")
        res = query_by_text(corpus, qtext, k=args.k)
        print(f"\n[Query file] {args.query_file}")
    else:
        # default: use first doc as query
        res = query_by_index(corpus, 0, k=args.k)
        print(f"\n[Query] 0 - {corpus.paths[0].split('/')[-1]}")

    print(f"Top {args.k} similar documents:")
    for (i, score) in res:
        print(f"  {i:>2}) {corpus.paths[i].split('/')[-1]:<30} similarity={score:.2f}")

if __name__ == "__main__":
    main()
