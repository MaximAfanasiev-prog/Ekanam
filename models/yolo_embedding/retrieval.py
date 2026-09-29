"""FAISS cosine search and writing submission artifacts in the format of dataset/README.md."""

import csv

import faiss
import numpy as np

from .metrics import confidence_text

TOP_K = 10


def search(query_emb, gallery_emb, k=TOP_K):
    """Exact cosine search (inner product over L2-normalized vectors). Returns (sims, idx), both Q x k."""
    index = faiss.IndexFlatIP(gallery_emb.shape[1])
    index.add(np.ascontiguousarray(gallery_emb, dtype=np.float32))
    return index.search(np.ascontiguousarray(query_emb, dtype=np.float32), min(k, len(gallery_emb)))


def top10(sims, k=TOP_K):
    """Like search, but from a precomputed similarity matrix: (sims, idx), both Q x k, ties by gallery order."""
    idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(sims, idx, axis=1), idx


def write_submission(path, query_ids, gallery_ids, idx):
    """submission.csv without a header, as the official evaluate.py expects: query_id,gallery_id_1..10."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for qid, row in zip(query_ids, idx, strict=False):
            w.writerow([qid, *(gallery_ids[j] for j in row)])


def write_candidates(path, query_ids, gallery_ids, sims, idx, threshold):
    """candidates.csv with a header: query_id,gallery_id,confidence.

    Every top-10 candidate with cosine similarity >= threshold is a row. A query whose best candidate is
    below the threshold gets no rows at all - this is the refusal (confirmed by the official evaluate.py).
    """
    n = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "gallery_id", "confidence"])
        for qid, s_row, i_row in zip(query_ids, sims, idx, strict=False):
            for s, j in zip(s_row, i_row, strict=False):
                if s >= threshold:
                    w.writerow([qid, gallery_ids[j], confidence_text(s)])
                    n += 1
    return n


def write_embeddings(path, query_emb, gallery_emb):
    """embeddings.npy: all test_query rows (file order), then all test_gallery rows (file order)."""
    np.save(path, np.concatenate([query_emb, gallery_emb]).astype(np.float32))
