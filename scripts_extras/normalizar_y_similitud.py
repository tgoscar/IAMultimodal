"""Demostracion: por que normalizar antes de comparar (pregunta en vivo tipica)."""
import numpy as np


def l2_normalize(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def cosine(u, v):
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(size=512)
    b = 10 * a + rng.normal(scale=0.1, size=512)  # misma direccion, magnitud 10x
    c = rng.normal(size=512)

    print("producto interno sin normalizar a.b:", round(float(a @ b), 2), " a.c:", round(float(a @ c), 2))
    an, bn, cn = l2_normalize(a), l2_normalize(b), l2_normalize(c)
    print("coseno (tras normalizar)        a.b:", round(float(an @ bn), 4), " a.c:", round(float(an @ cn), 4))
    print("=> sin normalizar, la magnitud domina el ranking; con L2, solo cuenta el angulo.")
