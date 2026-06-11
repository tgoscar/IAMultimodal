# Mejoras de código en vivo — Sección 9 del enunciado

Snippets listos para las 12 acciones que el docente puede pedir. Practícalos antes: la rúbrica da 3 pts a esto.

---

**1. Cambiar una consulta textual y explicar cómo cambia el ranking** (C9 §7)
```python
query = "a child playing with a dog"   # antes: "two men cooking in a kitchen"
qf = encode_texts(model, tokenizer, [query], device=device)
topk_text_to_image(qf, image_features, metadata, k=5)
```
*Explicar:* el embedding de la consulta se mueve en el espacio compartido → cambian los cosenos contra las mismas imágenes precomputadas. No se re-codifica ninguna imagen (ventaja dual encoder).

**2. Agregar un prompt alternativo para zero-shot** (C10 §4)
Editar `data/bootstrap_flickr30k/prompt_config.json` añadiendo `"a blurry photo of {}"` y relanzar `04_eval_zeroshot.py`. Comparar la nueva fila de `zeroshot_prompt_summary.csv`.

**3. Mostrar top-5 y no solo top-1**
```python
topk_image_to_text(img_idx, sim, caption_df, k=5)   # C10: cambiar k=1 → k=5
```

**4. Calcular similitud coseno entre embeddings**
```python
cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))
# si ya están normalizados L2: cos = u @ v
```

**5. Normalizar embeddings antes de comparar**
```python
feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
```
*Explicar:* sin normalizar, el producto interno mezcla ángulo y magnitud; el ranking puede sesgarse hacia vectores largos.

**6. Separar resultados correctos e incorrectos** (zero-shot)
```python
preds = pd.read_csv("outputs/metrics/zeroshot_predictions.csv")
aciertos = preds[preds["label"] == preds["pred"]]
errores  = preds[preds["label"] != preds["pred"]]
```

**7. Identificar un hard negative** (C9 §8)
```python
hard = mine_hard_negatives(sim, metadata, top_n=8)
hard.iloc[0]   # par incorrecto con score más alto
```
*Explicar:* score alto + etiqueta incorrecta = el modelo "casi" lo cree correcto; revela qué confunde (objetos compartidos, escena similar).

**8. Agregar una métrica simple como Recall@K**
```python
def recall_at_k(sim, k):
    # positivo correcto en la diagonal
    ranks = (-sim).argsort(axis=1)
    hits = [(ranks[i, :k] == i).any() for i in range(sim.shape[0])]
    return float(np.mean(hits))

for k in (1, 5, 10):
    print(f"R@{k} i2t = {recall_at_k(sim, k):.3f}   t2i = {recall_at_k(sim.T, k):.3f}")
```
(Versión multicaption en `scripts_extras/recall_at_k.py`.)

**9. Comparar dos checkpoints o configuraciones** (C10 §5)
```bash
python scripts/05_compare_checkpoints.py   # p. ej. ViT-B-32 laion2b vs openai
```
*Explicar:* mismo dataset, mismos prompts, solo cambian los pesos → diferencia atribuible al preentrenamiento.

**10. ¿Qué celda debe ejecutarse primero para reproducir el resultado?**
Orden: (1) `00_verify_env.py` → (2) `02_build_embeddings.py` (o la celda de C10 §2 que lo invoca si no existe el `.npz`) → (3) celda de la matriz `sim` → (4) métricas. Sin el `.npz`, todo lo demás falla.

**11. ¿Qué dependencia, versión o ruta podría romper la ejecución?**
- `open_clip_torch` ausente o versión incompatible con el nombre del checkpoint.
- Ruta `data/bootstrap_flickr30k/metadata.csv` relativa: el notebook exige ejecutarse desde la carpeta del proyecto (C9 lanza `RuntimeError` si no encuentra `src/`).
- CUDA/driver: sin GPU, cae a CPU (lento pero funciona); en C6, `num_workers` alto puede colgar en algunos entornos.
- Descarga de pesos: requiere red la primera vez (cache en `~/.cache`).

**12. ¿Qué parte del código corresponde al paper principal?**
- `encode_image` / `encode_text` → las dos torres de la Fig. 1 de CLIP.
- Normalización L2 + `image_features @ text_features.T` → similitud coseno de la pérdida contrastiva (en el paper, multiplicada por `logit_scale` = temperatura aprendida).
- Prompts "a photo of a {label}" → sección 3.1.4 del paper (prompt engineering y ensembling).
- El entrenamiento InfoNCE simétrico no lo re-ejecuto (entrenó con 400M de pares, WIT); lo represento a escala didáctica con el bi-encoder de C6.
