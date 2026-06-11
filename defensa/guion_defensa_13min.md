# Guion de defensa — 13 minutos

> Abiertos antes del turno: Cuadernos A, B y C **ejecutados** (con salidas visibles), este repo en GitHub,
> `trazabilidad.md`, y la carpeta `evidencias/`. Cámara activa + pantalla compartida.

## Bloque 1 — Paper y resultado principal (2 min)
"En la Evaluación 1 expuse **CLIP**, con la tesis de que el dual-encoder contrastivo es escalable para
zero-shot y retrieval imagen-texto. Para este examen construí **tres cuadernos propios** adaptando C6, C9 y C10
— cada uno declara en su cabecera qué celdas adaptó y qué cambió. Mi resultado principal: [⚠️ Recall@K
del Cuaderno A] con OpenCLIP ViT-B-32 sin ningún entrenamiento, y la brecha contra un bi-encoder entrenado
desde cero (Cuaderno C) que cuantifica el valor del preentrenamiento masivo."

## Bloque 2 — Cómo obtuve el resultado (3 min)
1. Cuaderno A (base C9+C6): embeddings normalizados → `sim = img @ txt.T` → Recall@{1,5,10} i2t/t2i; hard negatives; multicaption.
2. Cuaderno B (base C10): zero-shot construido con prompts, ensemble, checkpoints laion2b vs openai, búsqueda semántica.
3. Cuaderno C (base C6): la **InfoNCE de CLIP entrenada en vivo** + tabla comparativa contra CLIP + re-ranking del top-k.

## Bloque 3 — Verificación en vivo (3 min)
1. Cuaderno A → celda `# 3. CELDA CLAVE` → tabla `retrieval_recall.csv`.
2. Cuaderno A §5 → hard negative principal con su imagen.
3. Cuaderno B → `zeroshot_prompt_summary.csv` (único vs ensemble) y `checkpoint_comparison.csv`.
4. Cuaderno C → tabla `biencoder_vs_clip.csv` y `rerank_topk.csv`.
> Regla de la rúbrica: mostrar **la celda exacta** detrás de cada número (tope 14/20 si solo hay diapositivas).

## Bloque 4 — Pregunta técnica (3 min)
Material en `respuestas_clip_seccion_8_2.md` y `mejoras_codigo_en_vivo.md`. Si piden modificar:
narrar mientras editas (consulta nueva → solo se recodifica el texto; plantilla nueva → re-ejecutar la celda del ensemble).

## Bloque 5 — Defensa crítica (2 min)
Limitaciones: (1) subconjuntos pequeños → varianza en R@K, no afirmo significancia; (2) hard negatives
muestran fallas de composición — estructural del dual encoder; (3) etiquetas zero-shot por keyword son
ruidosas; (4) sesgos de datos web documentados en el propio paper (FairFace).
Mejora: escalar N + seeds; cross-encoder entrenado como re-ranker (la versión mínima ya está verificada en C §7).

## Checklist
- [ ] Tres cuadernos ejecutados con salidas (no celdas vacías) y subidos al repo
- [ ] `evidencias/` poblada automáticamente por los cuadernos
- [ ] ⚠️ Valores reales escritos en trazabilidad.md y en este guion
- [ ] Coherencia con los números de la diapositiva de la Evaluación 1
- [ ] Declaración de herramientas generativas en trazabilidad.md
- [ ] Prueba de pantalla compartida con los notebooks abiertos
