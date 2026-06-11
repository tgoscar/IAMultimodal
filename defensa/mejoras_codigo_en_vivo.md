# Mejoras de código en vivo — Sección 9 (referidas a MIS cuadernos)

**1. Cambiar una consulta textual y explicar el ranking** → Cuaderno A §4a: editar `query = "..."` y re-ejecutar.
*Explicar:* solo se recodifica el texto; las imágenes ya están indexadas — ventaja del dual encoder.

**2. Agregar un prompt alternativo para zero-shot** → Cuaderno B, `# 3. CELDA CLAVE`: añadir una plantilla a `TEMPLATES_ENSEMBLE` y re-ejecutar; comparar la nueva accuracy en la tabla.

**3. Mostrar top-5 y no solo top-1** → Cuaderno A: `topk_image_to_text(img_idx, k=5)` / `topk_text_to_image(query, k=5)` ya parametrizados; cambiar `k` en vivo.

**4. Calcular similitud coseno entre embeddings** →
```python
cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))   # si ya están normalizados: u @ v
```

**5. Normalizar embeddings antes de comparar** → señalar `f = f / f.norm(dim=-1, keepdim=True)` en `encode_images`/`encode_texts` (Cuaderno A celda 2) y el print de norma media ≈ 1.0.
*Explicar:* sin normalizar, el producto interno mezcla ángulo y magnitud y el ranking se sesga hacia vectores largos.

**6. Separar resultados correctos e incorrectos** → Cuaderno B: ya genera `aciertos` / `errores` y `zeroshot_errores.csv`.

**7. Identificar un hard negative** → Cuaderno A §5: `hard.iloc[0]` + su visualización.
*Explicar:* score alto + par incorrecto = qué confunde al modelo (objetos/escena compartidos).

**8. Agregar una métrica simple como Recall@K** → ya implementada (`recall_at_k`, A) y la variante multicaption (`recall_at_k_multicaption`, A §6); en vivo: añadir `R@3` a la tupla `KS`.

**9. Comparar dos checkpoints o configuraciones** → Cuaderno B §4 (`CONFIGS`); en vivo se puede añadir un tercer par `("ViT-B-16", "laion2b_s34b_b88k")` a la lista.

**10. ¿Qué celda debe ejecutarse primero?** → Cuaderno A celda 2 (genera `outputs/embeddings_flickr8k.npz`); B y C lo consumen (C lo verifica con `assert`). Sin ese `.npz`, B celda 1 y C celda 6 fallan.

**11. ¿Qué dependencia, versión o ruta podría romper la ejecución?**
- `open_clip_torch` ausente o nombre de checkpoint no soportado por la versión.
- Primera ejecución requiere red (descarga de `jxie/flickr8k` y de los pesos a `~/.cache`).
- `N_IMAGES` distinto entre A y C rompe el `assert` de la comparación.
- FAISS es opcional: el Cuaderno B cae a búsqueda exacta con numpy.

**12. ¿Qué parte del código corresponde al paper principal?**
- `encode_image`/`encode_text` (A) = las dos torres de la Fig. 1.
- Normalización L2 + `img @ txt.T` (A) = similitud coseno; en el paper va multiplicada por `logit_scale` (temperatura aprendida, init 1/0.07).
- `contrastive_loss` (C) = la InfoNCE simétrica del pseudocódigo de la Fig. 3 del paper.
- `TEMPLATES_ENSEMBLE` (B) = prompt engineering y ensembling, §3.1.4.
