# Hoja de trazabilidad — Examen Parcial Virtual MCC225A

> Sección 10 del enunciado. Lista **antes** del examen. ⚠️ = completar con tus valores reales tras ejecutar.

| Elemento | Respuesta |
|---|---|
| **Paper, modelo o línea temática asignada** | CLIP — *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021, arXiv:2103.00020) |
| **Cuadernos usados** | Base del curso: C6, C9, C10. Adaptados en tres cuadernos propios: A (embeddings + retrieval + hard negatives), B (zero-shot + prompts + checkpoints + búsqueda semántica), C (bi-encoder InfoNCE vs CLIP + re-ranking) |
| **Notebook, script o archivo ejecutado** | `notebooks/CuadernoA_clip_embeddings_retrieval.ipynb`, `CuadernoB_zeroshot_prompts_checkpoints.ipynb`, `CuadernoC_biencoder_vs_clip_reranking.ipynb` |
| **Celda, función o bloque de código clave** | A: `# 3. CELDA CLAVE` (`sim = image_features @ text_features.T` + `recall_at_k`). B: `# 3. CELDA CLAVE` (`class_embeddings` con ensemble y re-normalización). C: `contrastive_loss` (InfoNCE de CLIP) y `# 6. CELDA CLAVE` (tabla comparativa) |
| **Métrica, tabla, gráfico o evidencia usada** | `evidencias/metricas/retrieval_recall.csv`, `retrieval_recall_multicaption.csv`, `checkpoint_comparison.csv`, `biencoder_vs_clip.csv`, `rerank_topk.csv`; `zeroshot/zeroshot_prompt_summary.csv` + matriz de confusión; `hard_negatives/hard_negatives.csv` + imagen; galerías top-k |




