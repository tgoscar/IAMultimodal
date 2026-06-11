# Guion de defensa — 13 minutos (formato de la sección 5)

> Ten abiertos ANTES de tu turno: C9 ejecutado, C10 ejecutado, este repo, `trazabilidad.md`,
> y los CSVs de `outputs/metrics/`. Cámara activa + pantalla compartida.

---

## Bloque 1 — Recordatorio del paper y resultado principal (2 min)

"En la Evaluación 1 expuse **CLIP** (Radford et al., 2021), con la tesis de que el **dual-encoder contrastivo es una arquitectura escalable para transferencia zero-shot y recuperación imagen-texto**. CLIP entrena dos torres —un ViT para imagen y un transformer para texto— con una pérdida contrastiva simétrica (InfoNCE) sobre 400M de pares imagen-texto (WIT). Mi resultado principal fue [⚠️ tu evidencia: Recall@K de retrieval con OpenCLIP sobre el bootstrap de Flickr30k, y accuracy zero-shot con prompt ensemble]."

Una frase de cierre: "Hoy muestro cómo obtuve esos números con los cuadernos C6, C9 y C10."

## Bloque 2 — Cómo obtuve el resultado (3 min)

1. **C9:** cargué OpenCLIP `ViT-B-32 / laion2b_s34b_b79k`, generé embeddings normalizados con `02_build_embeddings.py` → `bootstrap_embeddings.npz`.
2. La similitud es `image_features @ text_features.T`; con pares alineados, la diagonal es el match correcto → `summarize_ranking` da Recall@K en i2t y t2i.
3. **C10:** zero-shot con `prompt_config.json` (plantillas + ensemble), comparación de checkpoints y FAISS.
4. **C6:** entrené el bi-encoder contrastivo a escala didáctica para mostrar la misma lógica de CLIP desde cero, y el cross-reranker como contraste arquitectónico.

## Bloque 3 — Verificación en vivo (3 min)

Plan de pantalla, en orden:
1. `Cuaderno9`: celda de carga del `.npz` → celda `sim = ...` → tabla Recall@K. *Señalar que coincide con la diapositiva de la Evaluación 1.*
2. Celda `mine_hard_negatives` → mostrar el primer hard negative con su imagen.
3. `Cuaderno10`: `zeroshot_prompt_summary.csv` (prompt único vs ensemble) y matriz de confusión.
4. Si hay tiempo: una consulta FAISS en vivo.

> Regla de oro de la rúbrica: **mostrar la celda exacta que generó cada número**. No mostrar solo diapositivas (tope 14/20).

## Bloque 4 — Pregunta técnica del docente (3 min)

Material preparado en:
- `respuestas_clip_seccion_8_2.md` (tus 5 preguntas asignadas),
- `mejoras_codigo_en_vivo.md` (las 12 acciones de código).

Si pide modificar algo: narrar mientras editas ("cambio la consulta, el embedding de texto se mueve en el espacio compartido, las imágenes ya están indexadas, por eso solo recalculo cosenos").

## Bloque 5 — Defensa crítica (2 min)

"Tres limitaciones honestas:
1. **Tamaño del bootstrap:** Recall@K con pocas imágenes tiene varianza alta; no afirmo significancia entre configuraciones.
2. **Composición:** los hard negatives muestran que el dual encoder empareja objetos y escena, no relaciones — limitación estructural de no tener interacción cruzada.
3. **Sesgos de datos web:** el propio paper documenta disparidades demográficas; el rendimiento zero-shot depende del diseño de clases.

Mejora propuesta: escalar al subconjunto 512/128/128 con `01_prepare_flickr30k_from_hf.py`, repetir con 3 seeds, y añadir **re-ranking cross-encoder (C6) sobre el top-10 del dual encoder**, midiendo el trade-off costo vs Recall@1."

---

## Checklist previo al examen

- [ ] Repo subido y URL en `trazabilidad.md`
- [ ] C6, C9, C10 ejecutados con salidas visibles (no celdas vacías)
- [ ] CSVs copiados a `evidencias/`
- [ ] Valores reales escritos donde hay ⚠️ (trazabilidad, guion, respuestas)
- [ ] Recall@K de la diapositiva de la Evaluación 1 = los del notebook (coherencia)
- [ ] Declaración de herramientas generativas escrita
- [ ] Probar compartir pantalla con el notebook abierto
