# Examen Parcial Virtual — MCC225A
## Radford et al., 2021 — arXiv:2103.00020

**Estudiante:** Oscar Benito Toledo Guerrero
**Cuadernos del curso usados como base:** C6, C9, C10
**Tesis (Evaluación 1):** el dual-encoder contrastivo es una arquitectura escalable para transferencia zero-shot y recuperación imagen-texto.



## Qué hace cada cuaderno

| Cuaderno | Base | Contenido | Evidencia que produce |
|---|---|---|---|
| **A — Embeddings y retrieval** | C9 + C6 | OpenCLIP `ViT-B-32/laion2b`, embeddings normalizados, matriz de similitud, **Recall@{1,5,10}** i2t/t2i, recuperación cruzada, **hard negatives**, retrieval **multicaption** (C10 §3) | `retrieval_recall.csv`, `hard_negatives.csv`, galerías top-k, `embeddings_flickr8k.npz` |
| **B — Zero-shot y prompts** | C10 | Clasificador zero-shot construido con prompts, **prompt único vs ensemble**, matriz de confusión, separación aciertos/errores, **comparación de checkpoints** (laion2b vs openai), búsqueda semántica (FAISS con fallback numpy) | `zeroshot_prompt_summary.csv`, `matriz_confusion.csv`, `checkpoint_comparison.csv`, `busqueda_semantica.csv` |
| **C — Bi-encoder vs CLIP** | C6 | **InfoNCE simétrica de CLIP entrenada en vivo** (bi-encoder didáctico), negativos semi-duros, tabla comparativa bi-encoder vs CLIP preentrenado (mismo test set), **re-ranking del top-k** | `biencoder_vs_clip.csv`, `rerank_topk.csv`, curva de pérdida |

## Reproducción

```bash
pip install -r requirements.txt

# Orden obligatorio: A genera los embeddings que B y C consumen
jupyter nbconvert --execute --to notebook --inplace notebooks/CuadernoA_clip_embeddings_retrieval.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/CuadernoB_zeroshot_prompts_checkpoints.ipynb
jupyter nbconvert --execute --to notebook --inplace notebooks/CuadernoC_biencoder_vs_clip_reranking.ipynb
```

Notas de entorno:
- Dataset: `jxie/flickr8k` vía HuggingFace (el mismo de C6); se descarga automáticamente la primera vez.
- Pesos: OpenCLIP descarga los checkpoints a `~/.cache` la primera vez (requiere red).
- GPU recomendada (contenedor Docker del curso). En CPU: bajar `N_IMAGES` en A/B y `MAX_TRAIN` en C.
- `N_IMAGES` (A y B) y `N_TEST` (C) deben coincidir — los cuadernos lo verifican con un `assert`.
- Todos los CSVs y PNGs de evidencia se escriben solos en `evidencias/`.

## Dónde está la celda clave de cada resultado

- **Recall@K del dual encoder:** Cuaderno A, celda marcada `# 3. CELDA CLAVE` (`sim = image_features @ text_features.T` + `recall_at_k`).
- **Zero-shot con prompts/ensemble:** Cuaderno B, celda `# 3. CELDA CLAVE` (`class_embeddings` con re-normalización tras el promedio).
- **InfoNCE = pérdida de CLIP:** Cuaderno C, función `contrastive_loss` (cross-entropy simétrica sobre la diagonal, temperatura 0.07).
- **Brecha de escala (tesis):** Cuaderno C, celda `# 6. CELDA CLAVE` (tabla bi-encoder didáctico vs CLIP).

## Limitación y mejora (resumen)

- **Limitación:** subconjuntos pequeños → varianza alta en Recall@K; etiquetas zero-shot derivadas por keyword son ruidosas; el bi-encoder didáctico está sub-entrenado por diseño.
- **Mejora:** escalar N y repetir con seeds; re-ranking con cross-encoder entrenado (C6 §9-13) sobre el top-k; ensembles de prompts específicos por dominio. El Cuaderno C ya verifica la versión mínima del re-ranking.
