# Respuestas preparadas — Preguntas generales (sección 7 del enunciado)

> ⚠️ Donde aparece `___`, completa con tus valores reales tras ejecutar los cuadernos.

**1. ¿Qué resultado concreto de la Evaluación 1 puede reproducir o verificar en vivo?**
La tabla de Recall@K de recuperación cruzada con OpenCLIP `ViT-B-32 / laion2b_s34b_b79k` sobre el bootstrap de Flickr30k (slide de evidencia de mi presentación). En vivo: cargo `outputs/embeddings/bootstrap_embeddings.npz`, ejecuto la celda de `sim = image_features @ text_features.T` + `summarize_ranking` en C9 y muestro que reproduce los valores de la diapositiva (R@1=___, R@5=___, R@10=___). También puedo verificar la accuracy zero-shot con/sin prompt ensemble (C10 §4).

**2. ¿Qué cuaderno del curso usó directamente y qué parte del código adaptó?**
C9 como baseline (lo ejecuté casi íntegro), C10 para zero-shot/prompts/checkpoints/FAISS, y C6 para el contraste conceptual bi-encoder vs cross-encoder. Adapté: plantillas de prompt propias en `prompt_config.json`, una consulta textual nueva para FAISS, y el script `scripts_extras/recall_at_k.py` para reportar K configurables.

**3. ¿Qué celdas, funciones o scripts generaron sus resultados principales?**
`scripts/02_build_embeddings.py` (embeddings), C9 §6 (`summarize_ranking` → Recall@K), C9 §8 (`mine_hard_negatives`), `scripts/04_eval_zeroshot.py` (accuracy y matriz de confusión), `scripts/05_compare_checkpoints.py` (tabla de checkpoints).

**4. ¿Qué cambió respecto al cuaderno original?**
⚠️ Declara tus cambios reales. Sugeridos en este repo: plantillas extra en el ensemble, K adicionales en Recall@K, segunda configuración de checkpoint, consulta FAISS propia, y separación de aciertos/errores en el zero-shot.

**5. ¿Qué métrica, tabla, gráfico, ranking o evidencia usó?**
Recall@{1,5,10} en ambas direcciones (tabla), accuracy zero-shot por prompt y ensemble (tabla + matriz de confusión), tabla de hard negatives con scores, galería top-k de recuperación (gráfico), `checkpoint_comparison.csv`.

**6. ¿Qué error, limitación o sesgo observó?**
(a) Bootstrap pequeño → alta varianza, resultados no generalizables. (b) Hard negatives revelan que el modelo empareja por objetos/escena, no por composición. (c) Zero-shot sensible a la redacción del prompt (sesgo de plantilla). (d) Sesgo de datos web en LAION/WIT: el propio paper documenta disparidades (FairFace) — el rendimiento depende del diseño de clases.

**7. ¿Qué pasaría si cambia el modelo, el prompt, el checkpoint, el dataset o los negativos?**
- *Modelo/checkpoint:* `05_compare_checkpoints.py` lo mide directamente; checkpoints más grandes o con más datos suelen subir R@K, pero con más costo de inferencia.
- *Prompt:* cambia la accuracy zero-shot sin tocar pesos — esa es la gracia del zero-shot por lenguaje natural; el ensemble estabiliza.
- *Dataset:* pasar del bootstrap al subconjunto 512/128/128 baja los valores absolutos (más distractores) pero da estimaciones más fiables.
- *Negativos:* en entrenamiento (C6), pasar de aleatorios a semi-duros hace la pérdida más informativa y mejora el ranking fino; en evaluación, más candidatos = tarea más difícil.

**8. ¿Qué parte de su implementación corresponde a conceptos desarrollados en clase?**
Aprendizaje contrastivo InfoNCE y bi-encoders (C6), embeddings compartidos imagen-texto y retrieval con ranking (C6/C9), hard negatives (C6/C9), zero-shot y prompt ensembles (C10), reproducibilidad con scripts/Docker/SLURM (C9), búsqueda semántica con FAISS (C10).

**9. ¿Qué mejora haría para convertir el experimento en un pipeline reproducible?**
Ya parto del pipeline de C9 (scripts numerados + configs YAML). Mejoras: fijar seeds y versiones en `requirements-extra.txt` con hashes, registrar cada corrida (modelo, checkpoint, dataset, fecha, git commit) en un `runs.csv`, un `Makefile`/`run_local_pipeline.sh` único de extremo a extremo, y tests mínimos (shape y norma de los embeddings, diagonal de `sim` > media).

**10. ¿Qué resultado no puede asegurar todavía y por qué?**
Que las diferencias entre prompts/checkpoints sean estadísticamente significativas: con un bootstrap de pocas decenas de imágenes el intervalo de confianza de Recall@K es muy ancho. Tampoco puedo asegurar generalización fuera de Flickr30k (distinta distribución de dominio). Por eso mi mejora propuesta es escalar el subconjunto y repetir con varios seeds.
