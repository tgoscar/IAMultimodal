"""Prompt ensemble propio para zero-shot (modificacion sobre C10, seccion 4).

Uso (desde la carpeta del proyecto del curso, con el entorno Docker activo):
    python scripts_extras/prompt_ensemble.py
Requiere: open_clip_torch y el .npz de embeddings ya construido (02_build_embeddings.py).
"""
import json
import numpy as np

# Plantillas: las del curso + plantillas propias añadidas
TEMPLATES = [
    "a photo of a {}",
    "a photo of {}",
    "a photo of people {}",          # propia
    "a low resolution photo of {}",  # propia
    "a close-up photo of {}",        # propia
]


def class_embeddings(model, tokenizer, labels, device, templates=TEMPLATES):
    """Un embedding por clase: promedio normalizado de todas sus plantillas."""
    import torch
    embs = []
    with torch.no_grad():
        for label in labels:
            texts = [t.format(label) for t in templates]
            tokens = tokenizer(texts).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # normalizar cada plantilla
            emb = feats.mean(dim=0)
            emb = emb / emb.norm()                            # re-normalizar tras promediar (clave)
            embs.append(emb.cpu().numpy())
    return np.stack(embs)


def zeroshot_accuracy(image_features, class_embs, true_label_idx):
    """image_features: (N, d) normalizados. class_embs: (C, d). true_label_idx: (N,)."""
    logits = image_features @ class_embs.T
    preds = logits.argmax(axis=1)
    return float((preds == np.asarray(true_label_idx)).mean())


if __name__ == "__main__":
    import open_clip, torch, pandas as pd
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    bundle = np.load("outputs/embeddings/bootstrap_embeddings.npz", allow_pickle=True)
    image_features = bundle["image_features"]
    metadata = pd.read_csv("data/bootstrap_flickr30k/metadata.csv")
    labels = sorted(metadata["label"].unique())
    true_idx = metadata["label"].map({l: i for i, l in enumerate(labels)}).values

    # Comparacion: prompt unico vs ensemble
    single = class_embeddings(model, tokenizer, labels, device, templates=["a photo of a {}"])
    ens = class_embeddings(model, tokenizer, labels, device)
    print(f"accuracy prompt unico : {zeroshot_accuracy(image_features, single, true_idx):.4f}")
    print(f"accuracy ensemble ({len(TEMPLATES)} plantillas): {zeroshot_accuracy(image_features, ens, true_idx):.4f}")
