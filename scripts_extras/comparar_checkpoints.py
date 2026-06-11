"""Comparar dos checkpoints de OpenCLIP con las mismas consultas (extiende C10, seccion 5).

Mide Recall@K sobre el mismo metadata.csv para dos configuraciones y guarda un CSV.
"""
import numpy as np
import pandas as pd

CONFIGS = [
    ("ViT-B-32", "laion2b_s34b_b79k"),
    ("ViT-B-32", "openai"),          # segundo checkpoint: pesos originales de OpenAI
]


def build_embeddings(model_name, pretrained, filepaths, captions, device):
    import torch, open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    with torch.no_grad():
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in filepaths]).to(device)
        img_f = model.encode_image(imgs)
        txt_f = model.encode_text(tokenizer(captions).to(device))
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return img_f.cpu().numpy(), txt_f.cpu().numpy()


def recall_at_k(sim, k):
    ranks = (-sim).argsort(axis=1)
    return float(np.mean([(ranks[i, :k] == i).any() for i in range(sim.shape[0])]))


if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = pd.read_csv("data/bootstrap_flickr30k/metadata.csv")
    rows = []
    for model_name, pretrained in CONFIGS:
        img_f, txt_f = build_embeddings(model_name, pretrained,
                                        meta["filepath"].tolist(), meta["caption"].tolist(), device)
        sim = img_f @ txt_f.T
        rows.append({"model": model_name, "pretrained": pretrained,
                     **{f"i2t_R@{k}": recall_at_k(sim, k) for k in (1, 5, 10)},
                     **{f"t2i_R@{k}": recall_at_k(sim.T, k) for k in (1, 5, 10)}})
    df = pd.DataFrame(rows)
    df.to_csv("evidencias/metricas/checkpoint_comparison_extra.csv", index=False)
    print(df.to_string(index=False))
