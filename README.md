# Examen Parcial Virtual — MCC225A
## Defensa técnica: CLIP (Radford et al., 2021 — arXiv:2103.00020)

**Estudiante:** Oscar Benito Toledo Guerrero
**Tema asignado:** CLIP — Contrastive Language-Image Pre-training
**Cuadernos del curso utilizados:** C6, C9, C10
**Tesis defendida en la Evaluación 1:** el dual-encoder contrastivo es una arquitectura escalable para transferencia zero-shot y recuperación imagen-texto.

---

## 1. Contenido del repositorio

```
repo-examen-parcial-clip/
├── README.md                  ← este archivo (instrucciones de reproducción)
├── trazabilidad.md            ← hoja de trazabilidad exigida en la sección 10 del examen
├── notebooks/
│   ├── Cuaderno6-MCC225.ipynb   ← bi-encoder contrastivo + re-ranking (Flickr8k)
│   ├── Cuaderno9-MCC225.ipynb   ← OpenCLIP baseline, retrieval, hard negatives (Flickr30k bootstrap)
│   └── Cuaderno10-MCC225.ipynb  ← zero-shot, prompt ensembles, checkpoints, FAISS
├── evidencias/
│   ├── metricas/        ← CSVs de Recall@K, accuracy zero-shot, comparación de checkpoints
│   ├── graficos/        ← curvas de pérdida, galerías de retrieval, matriz de confusión
│   ├── rankings/        ← top-k de recuperación cruzada (i2t y t2i)
│   ├── hard_negatives/  ← pares incorrectos con score alto (salida de mine_hard_negatives)
│   └── zeroshot/        ← predicciones y resumen por prompt/ensemble
├── scripts_extras/      ← modificaciones propias sobre los cuadernos originales
│   ├── prompt_ensemble.py
│   ├── recall_at_k.py
│   ├── comparar_checkpoints.py
│   └── normalizar_y_similitud.py
└── defensa/
    ├── guion_defensa_13min.md
    ├── respuestas_clip_seccion_8_2.md
    ├── respuestas_preguntas_generales.md
    └── mejoras_codigo_en_vivo.md
```

> **Importante antes del examen:** las carpetas de `evidencias/` deben llenarse con
> **tus salidas reales** al ejecutar los cuadernos (CSVs de `outputs/metrics/`,
> capturas de gráficos, etc.). Los archivos `RESULTADOS_AQUI.md` indican qué va en cada una.

---

## 2. Entorno y dependencias

Los cuadernos C9 y C10 están diseñados para el contenedor Docker del curso con una GPU local.

```bash
# Dentro del contenedor del curso
pip install -r requirements-extra.txt        # open_clip_torch, faiss-cpu, etc.
python scripts/00_verify_env.py              # verifica PyTorch, CUDA, GPU y open_clip
```

Dependencias clave: `torch`, `open_clip_torch`, `transformers`, `datasets`, `faiss`, `pandas`, `matplotlib`.

**Checkpoint baseline:** `ViT-B-32` con pesos `laion2b_s34b_b79k` (definido en `configs/local.yaml`).

---

## 3. Reproducción mínima (orden de ejecución)

### Paso 1 — C9: baseline OpenCLIP sobre el bootstrap de Flickr30k
```bash
cd Semana4/
python scripts/02_build_embeddings.py \
  --metadata-csv data/bootstrap_flickr30k/metadata.csv \
  --model-name ViT-B-32 --pretrained laion2b_s34b_b79k
```
Esto genera `outputs/embeddings/bootstrap_embeddings.npz` (embeddings de imagen y texto **ya normalizados L2**).
Luego, en el notebook `Cuaderno9-MCC225.ipynb`:
- la celda `sim = image_features @ text_features.T` produce la **matriz de similitud**,
- `summarize_ranking(sim)` y `summarize_ranking(sim.T)` producen **Recall@K** en ambas direcciones (i2t y t2i),
- `mine_hard_negatives(sim, metadata, top_n=8)` produce la evidencia de **negativos duros**.

### Paso 2 — C10: zero-shot, prompt ensembles y comparación de checkpoints
```bash
python scripts/04_eval_zeroshot.py \
  --embeddings outputs/embeddings/bootstrap_embeddings.npz \
  --metadata-csv data/bootstrap_flickr30k/metadata.csv \
  --prompt-config data/bootstrap_flickr30k/prompt_config.json

python scripts/05_compare_checkpoints.py     # genera outputs/metrics/checkpoint_comparison.csv
python scripts/06_build_faiss_index.py       # genera outputs/faiss/query_results.csv
```
El notebook `Cuaderno10-MCC225.ipynb` consume estos outputs: retrieval **multicaption**
(`compute_similarity_matrix` + métricas i2t/t2i), resumen zero-shot por prompt, matriz de confusión y búsqueda FAISS.

### Paso 3 — C6: bi-encoder entrenado desde cero + re-ranker (contraste conceptual con CLIP)
Ejecutar `Cuaderno6-MCC225.ipynb` de inicio a fin (Flickr8k vía HuggingFace `jxie/flickr8k`;
en CPU usar `fast_dev_run` y subconjuntos reducidos según la sección "Orientación de uso").
Produce: métricas del bi-encoder contrastivo (InfoNCE), métricas del cross-reranker y la tabla comparativa final.

### Paso 4 — Escalar más allá del bootstrap (opcional, mostrado como mejora)
```bash
python scripts/01_prepare_flickr30k_from_hf.py \
  --output-root data/processed/flickr30k_hf \
  --train-limit 512 --val-limit 128 --test-limit 128
```
y repetir Pasos 1–2 cambiando el CSV de entrada.

---

## 4. Relación resultado ↔ paper ↔ cuadernos

| Concepto del paper CLIP | Dónde se verifica en el código |
|---|---|
| Dual encoder (dos torres independientes) | C6: `TextTower`/torre visual del `ContrastiveBiEncoder`; C9: `model.encode_image` / `encode_texts` de OpenCLIP |
| Pérdida contrastiva InfoNCE (simétrica) | C6: entrenamiento del bi-encoder (`train_biencoder`) |
| Similitud coseno como producto interno de embeddings normalizados | C9 celda `sim = image_features @ text_features.T`; C10 `compute_similarity_matrix` |
| Zero-shot vía prompts ("a photo of a {label}") | C10 sección 4 + `prompt_config.json` + `04_eval_zeroshot.py` |
| Prompt engineering / ensembles | C10: `zeroshot_prompt_summary.csv` compara plantillas y ensemble |
| Escala y checkpoints (WIT → LAION) | C10 sección 5: `05_compare_checkpoints.py` |
| Limitación del dual encoder en interacción fina | C6: re-ranker cross-encoder mejora el ranking top-k del bi-encoder |
| Retrieval como sistema real | C10 sección 6: índice FAISS |

---

## 5. Limitación encontrada y mejora propuesta (resumen)

- **Limitación:** las métricas se calculan sobre un *bootstrap* pequeño de Flickr30k; los intervalos de
  confianza de Recall@K son amplios y los hard negatives muestran que el dual encoder no discrimina
  composición fina (orden sujeto-objeto, conteo).
- **Mejora:** (1) escalar al subconjunto de 512/128/128 con `01_prepare_flickr30k_from_hf.py` y reportar
  Recall@K con desviación entre seeds; (2) añadir re-ranking cross-encoder (C6) sobre el top-k del
  dual encoder, midiendo el trade-off costo/Recall@1.

Detalle completo en `defensa/` y en `trazabilidad.md`.
