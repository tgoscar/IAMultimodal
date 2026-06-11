# Respuestas preparadas — Sección 8.2 (Oscar Benito Toledo Guerrero, Tema: CLIP)

---

## 1. ¿Cómo obtuvo los embeddings de imagen y texto?

**Respuesta corta:** con OpenCLIP (`ViT-B-32`, checkpoint `laion2b_s34b_b79k`), usando las dos torres del modelo por separado, y guardándolos normalizados en un `.npz`.

**Desarrollo (mostrar en pantalla):**
- En **C9**, la celda de la sección 5 llama a `encode_image_paths(model, preprocess, metadata["filepath"], ...)`:
  cada imagen pasa por el `preprocess` de torchvision (resize 224, center crop, normalización con las medias/std de CLIP) y luego por `model.encode_image(...)`, que es el **Vision Transformer**.
- Los textos pasan por `encode_texts(model, tokenizer, captions, ...)`: el tokenizador BPE de CLIP (contexto de 77 tokens) y `model.encode_text(...)`, el **transformer de texto**, que toma la representación del token EOT y la proyecta con `text_projection`.
- Ambos vectores se **normalizan L2** antes de guardarse — esto es clave porque convierte el producto interno en similitud coseno.
- Fuera del notebook, lo mismo se materializa con `scripts/02_build_embeddings.py`, que produce `outputs/embeddings/bootstrap_embeddings.npz`. Eso hace el pipeline reproducible: el notebook solo *consume* el `.npz`.

**Conexión con el paper:** es exactamente el flujo de la Figura 1 de CLIP — dos encoders independientes que proyectan a un espacio compartido de dimensión `embed_dim` (512 para ViT-B/32).

---

## 2. ¿Dónde se calcula la similitud entre imagen y texto?

**Respuesta corta:** en una sola línea: `sim = image_features @ text_features.T` (C9, sección 6) o su equivalente `compute_similarity_matrix(image_features, text_features)` en C10.

**Desarrollo:**
- Como los embeddings ya están normalizados, `u @ v.T` = cos(θ). La matriz `sim` es de tamaño N_imágenes × N_textos; **cada fila** es el ranking de textos para una imagen (image→text) y **cada columna**, transponiendo, el ranking de imágenes para un texto (text→image).
- En el bootstrap los pares están alineados, así que **la diagonal es el match correcto**: `summarize_ranking(sim)` mide en qué posición del ranking cae la diagonal → Recall@K.
- En el CLIP original (`clip/model.py`, método `forward`), la misma operación aparece como
  `logits_per_image = logit_scale.exp() * image_features @ text_features.t()` — el factor extra es la **temperatura aprendida** (inicializada en 1/0.07), que escala los logits antes del softmax de la pérdida contrastiva. En inferencia para ranking, la temperatura no cambia el orden, por eso en los cuadernos se omite.

---

## 3. ¿Qué significa que CLIP funcione como dual encoder?

**Respuesta corta:** que imagen y texto se codifican por **torres separadas que nunca interactúan** hasta el producto punto final.

**Desarrollo:**
- No hay atención cruzada entre tokens visuales y textuales: la única "interacción" es un escalar, la similitud coseno. Esto contrasta con un **cross-encoder** (como el `LightCrossReranker` de C6), donde imagen y texto se concatenan y se procesan juntos.
- **Consecuencia práctica que demuestro en mi código:** puedo **precomputar e indexar** los embeddings de las imágenes una sola vez (el `.npz`, y en C10 §6 el índice **FAISS**) y responder cualquier consulta nueva con un solo `encode_text` + búsqueda de vecinos. Eso da complejidad O(N+M) de codificación en vez de O(N×M) pases del modelo — y es lo que hace a CLIP **escalable** para retrieval, que es justamente mi tesis de la Evaluación 1.
- **El costo:** sin interacción fina, el modelo confunde composiciones ("hombre muerde perro" vs "perro muerde hombre") — lo muestro con los **hard negatives** de C9 §8 y lo conecto con el re-ranker de C6 como solución híbrida (recuperar con bi-encoder, refinar top-k con cross-encoder).

---

## 4. ¿Qué cambiaría en el código para probar varios prompts por clase?

**Respuesta corta:** ya lo hace C10 con `prompt_config.json` + `04_eval_zeroshot.py`; mi modificación añade plantillas propias y promedia los embeddings de texto por clase (**prompt ensemble**).

**Desarrollo (mostrar `scripts_extras/prompt_ensemble.py`):**
```python
templates = [
    "a photo of a {}",
    "a photo of people {}",          # plantilla propia
    "a low resolution photo of {}",  # plantilla propia
]

def class_embedding(model, tokenizer, label, templates, device):
    texts = [t.format(label) for t in templates]
    feats = encode_texts(model, tokenizer, texts, device=device)  # (T, d), ya normalizados
    emb = feats.mean(axis=0)                # promedio de las T plantillas
    return emb / np.linalg.norm(emb)        # re-normalizar: clave tras promediar
```
- Un embedding por clase = promedio normalizado de los embeddings de todas sus plantillas. Después la clasificación zero-shot es el mismo `image_features @ class_embeddings.T`.
- **Qué esperar:** el paper reporta ~+3.5% en ImageNet con ensemble de 80 prompts; en mi bootstrap el efecto se ve en `zeroshot_prompt_summary.csv`, comparando prompt único vs ensemble.
- **Detalle técnico defendible:** hay que re-normalizar **después** de promediar, porque el promedio de vectores unitarios no es unitario.

---

## 5. Si mejora Recall@5 pero no Recall@1, ¿cómo interpreta ese resultado?

**Respuesta corta:** el modelo coloca al positivo correcto **dentro del vecindario top-5 con más frecuencia**, pero **no logra distinguirlo de sus vecinos más duros** para ponerlo primero. Mejora el recall grueso, no la discriminación fina.

**Desarrollo:**
1. **Diagnóstico:** Recall@1 mide ranking fino; Recall@5 mide si el positivo entra en la vecindad semántica. Si solo sube R@5, el cambio (prompt, checkpoint, más datos) está acercando el positivo al cluster correcto, pero en el top del ranking quedan **hard negatives** casi indistinguibles.
2. **Causa típica en mi setup:** en Flickr30k cada imagen tiene 5 captions casi parafraseadas (retrieval **multicaption**, C10 §3). Captions de imágenes distintas pero de la misma escena ("a dog running on grass") compiten en el top-1; el dual encoder, sin interacción cruzada, no resuelve esos matices.
3. **Verificación que mostraría:** la tabla de `mine_hard_negatives` — los errores top-1 comparten objetos y escena con el positivo, no son errores aleatorios.
4. **Acción:** (a) **re-ranking con cross-encoder** (C6) sobre el top-5/top-10 del bi-encoder: ataca exactamente R@1 sin encarecer la recuperación; (b) prompts/consultas más específicas; (c) si fuera entrenamiento propio, negativos semi-duros en el batch (C6 §3).
5. **Lectura honesta:** en un bootstrap pequeño, una diferencia de R@1 puede ser ruido — por eso propongo escalar el subconjunto y repetir con varios seeds antes de afirmar mejora.
