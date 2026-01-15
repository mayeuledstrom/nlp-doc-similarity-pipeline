from src.similarity_pipeline import build_corpus, fit_tfidf, query_by_index
import os

def test_pipeline_runs(tmp_path):
    # Prepare small temp corpus
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("machine learning for documents similarity", encoding="utf-8")
    (d / "b.txt").write_text("documents similarity with cosine distance", encoding="utf-8")
    (d / "c.txt").write_text("energy management in smart buildings", encoding="utf-8")

    corpus = build_corpus(str(d))
    corpus = fit_tfidf(corpus)
    res = query_by_index(corpus, 0, k=2)
    assert len(res) == 2
