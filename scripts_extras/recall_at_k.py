"""Recall@K parametrizable (modificacion sobre C9/C10).

Dos variantes:
- recall_at_k_diagonal: pares alineados 1 a 1 (bootstrap de C9, diagonal = positivo)
- recall_at_k_multicaption: una imagen tiene varias captions positivas (C10, seccion 3)
"""
import numpy as np


def recall_at_k_diagonal(sim, k):
    """sim: (N, N), positivo en la diagonal."""
    ranks = (-sim).argsort(axis=1)
    hits = [(ranks[i, :k] == i).any() for i in range(sim.shape[0])]
    return float(np.mean(hits))


def recall_at_k_multicaption(sim, positives, k):
    """sim: (Q, C). positives: dict {query_idx: [indices positivos]}."""
    ranks = (-sim).argsort(axis=1)
    hits = []
    for q, pos in positives.items():
        topk = set(ranks[q, :k].tolist())
        hits.append(len(topk & set(pos)) > 0)
    return float(np.mean(hits))


if __name__ == "__main__":
    bundle = np.load("outputs/embeddings/bootstrap_embeddings.npz", allow_pickle=True)
    img, txt = bundle["image_features"], bundle["text_features"]
    sim = img @ txt.T
    print("direccion      ", "  ".join(f"R@{k:<3}" for k in (1, 5, 10)))
    print("image -> text  ", "  ".join(f"{recall_at_k_diagonal(sim, k):.3f}" for k in (1, 5, 10)))
    print("text  -> image ", "  ".join(f"{recall_at_k_diagonal(sim.T, k):.3f}" for k in (1, 5, 10)))
