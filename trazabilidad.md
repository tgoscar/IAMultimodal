# Hoja de trazabilidad — Examen Parcial Virtual MCC225A

> Exigida en la **sección 10** del enunciado. Debe estar lista **antes** del inicio del examen.
> Los campos marcados con ⚠️ debes completarlos con tus valores reales después de ejecutar los cuadernos.

| Elemento | Respuesta |
|---|---|
| **Paper, modelo o línea temática asignada** | CLIP — *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., 2021, arXiv:2103.00020) |
| **URL del repositorio entregado** | ⚠️ `https://github.com/<tu-usuario>/repo-examen-parcial-clip` |
| **Cuadernos usados** | C6 (bi-encoder contrastivo + re-ranking), C9 (OpenCLIP baseline sobre bootstrap Flickr30k), C10 (zero-shot, prompts, checkpoints, FAISS) |
| **Notebook, script o archivo ejecutado** | `Cuaderno9-MCC225.ipynb` y `Cuaderno10-MCC225.ipynb` (principales); `scripts/02_build_embeddings.py`, `scripts/04_eval_zeroshot.py`, `scripts/05_compare_checkpoints.py`; `Cuaderno6-MCC225.ipynb` (contraste bi-encoder vs cross-encoder) |
| **Celda, función o bloque de código clave** | C9 §6: `sim = image_features @ text_features.T` + `summarize_ranking(sim)` (Recall@K i2t/t2i). C10 §3: `compute_similarity_matrix` con evaluación multicaption. C10 §4: zero-shot con `prompt_config.json` |
| **Resultado obtenido** | ⚠️ Recall@1/@5/@10 image→text = ___ / ___ / ___ ; text→image = ___ / ___ / ___ (ViT-B-32, laion2b_s34b_b79k, bootstrap Flickr30k). Accuracy zero-shot: prompt simple = ___ , ensemble = ___ |
| **Métrica, tabla, gráfico o evidencia usada** | Tabla de Recall@K (`evidencias/metricas/`), tabla de hard negatives (`mine_hard_negatives`), matriz de confusión zero-shot, `checkpoint_comparison.csv`, ejemplos top-k de recuperación cruzada |
| **Cambio hecho sobre el cuaderno original** | ⚠️ Ejemplos preparados en `scripts_extras/`: (1) prompt ensemble propio con plantillas adicionales; (2) función `recall_at_k` parametrizable; (3) comparación de un segundo checkpoint; (4) consulta textual nueva sobre el índice FAISS. *Declara aquí los que realmente hiciste.* |
| **Limitación encontrada** | Bootstrap pequeño → varianza alta en Recall@K; el dual encoder falla en composición fina (hard negatives con score alto comparten objetos pero no relaciones); accuracy zero-shot sensible a la redacción del prompt |
| **Mejora propuesta** | Escalar a subconjunto 512/128/128 de Flickr30k (`01_prepare_flickr30k_from_hf.py`) + re-ranking con cross-encoder (C6) sobre el top-10 del dual encoder; reportar media ± desviación con 3 seeds |

## Declaración de herramientas

⚠️ El enunciado penaliza (1–3 pts) el **uso no declarado de herramientas generativas**. Declara aquí
cualquier asistencia de IA usada en la preparación del material (por ejemplo: "se utilizó un asistente
de IA para organizar el repositorio y redactar borradores de respuestas; todo el código fue ejecutado
y verificado personalmente").
