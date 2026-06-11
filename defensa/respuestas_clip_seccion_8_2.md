# Respuestas preparadas — Sección 8.2 (Tema: CLIP) — referidas a MIS cuadernos

## 1. ¿Cómo obtuvo los embeddings de imagen y texto?

Con OpenCLIP (`ViT-B-32`, checkpoint `laion2b_s34b_b79k`), en mi **Cuaderno A, celda 2** (`encode_images` / `encode_texts`, adaptadas de C9 §4-5, que en el curso vivían en `src/`):
- cada imagen pasa por el `preprocess` de CLIP (resize 224, center crop, normalización con medias/std de CLIP) y por `model.encode_image` → torre visual ViT;
- cada texto pasa por el tokenizador BPE (contexto 77) y `model.encode_text` → torre textual transformer (toma el token EOT y proyecta con `text_projection`);
- **ambos se normalizan L2** antes de guardarse en `outputs/embeddings_flickr8k.npz` — eso convierte el producto interno en coseno.

*Mostrar en pantalla:* la celda 2 del Cuaderno A y el print de la norma media ≈ 1.0.

## 2. ¿Dónde se calcula la similitud entre imagen y texto?

**Cuaderno A, celda marcada `# 3. CELDA CLAVE`:** `sim = image_features @ text_features.T`.
- Embeddings normalizados ⇒ `u @ v.T = cos(θ)`. La matriz es N×N; cada fila = ranking de textos para una imagen; cada columna (transponiendo) = ranking de imágenes para un texto. Pares alineados ⇒ **la diagonal es el positivo**, y `recall_at_k` mide si la diagonal cae en el top-k.
- En el CLIP original (`model.py`, `forward`) la misma operación es `logit_scale.exp() * image_features @ text_features.t()`: el factor extra es la **temperatura aprendida** (init 1/0.07) que escala los logits para la pérdida; en inferencia no cambia el orden del ranking, por eso la omito.

## 3. ¿Qué significa que CLIP funcione como dual encoder?

Que imagen y texto se codifican por **torres separadas que nunca interactúan** hasta el producto punto final — no hay atención cruzada entre tokens visuales y textuales.
- **Lo demuestro con mis tres cuadernos:** en A, las imágenes se codifican **una sola vez** y cualquier consulta nueva cuesta un solo `encode_text` (§4: "cambio la consulta y solo recalculo cosenos"); en B, eso permite **indexar con FAISS**; en C, implemento el dual encoder desde cero (`ContrastiveBiEncoder`) y discuto la alternativa cross-encoder.
- **El costo:** sin interacción fina, falla en composición — lo evidencio con los **hard negatives** del Cuaderno A §5 (los errores comparten objetos/escena con el positivo, difieren en relaciones).

## 4. ¿Qué cambiaría en el código para probar varios prompts por clase?

Ya está implementado en mi **Cuaderno B, celda `# 3. CELDA CLAVE`** (adaptado de C10 §4):
```python
TEMPLATES_ENSEMBLE = ["a photo of a {}", "a photo of {}", "a photo of people with a {}",
                      "a low resolution photo of {}", "a close-up photo of {}"]
# por clase: codificar todas las plantillas, normalizar, PROMEDIAR y RE-normalizar
emb = feats.mean(dim=0);  emb = emb / emb.norm()
```
- Detalle defendible: hay que **re-normalizar después de promediar** (el promedio de vectores unitarios no es unitario).
- El cuaderno produce la tabla prompt único vs ensemble (`zeroshot_prompt_summary.csv`). El paper reporta ~+3.5% en ImageNet con ensemble de 80 prompts (§3.1.4).
- *Cambio en vivo si lo piden:* añadir una plantilla a la lista y re-ejecutar la celda.

## 5. Si mejora Recall@5 pero no Recall@1, ¿cómo interpreta ese resultado?

El positivo entra más a menudo al **vecindario top-5**, pero el modelo **no lo distingue de sus vecinos duros** para ponerlo primero: mejora el recall grueso, no la discriminación fina.
1. **Causa típica:** captions casi parafraseadas compiten en el top-1 (lo mido con retrieval **multicaption**, Cuaderno A §6) y hard negatives que comparten escena (Cuaderno A §5).
2. **Verificación en mis propios datos:** Cuaderno C §7 — re-rankear el top-10 del bi-encoder con un scorer más fuerte **sube R@1 sin tocar el techo R@10**: el cuello de botella está en el orden dentro del top-k, exactamente este fenómeno.
3. **Acción:** re-ranking con cross-encoder (C6 §9-13) sobre el top-k; prompts/consultas más específicos; en entrenamiento, negativos semi-duros (Cuaderno C §2).
4. **Lectura honesta:** con subconjuntos pequeños una diferencia de R@1 puede ser ruido; antes de afirmar mejora, escalar N y repetir con varios seeds.
