"""Builds a small product catalog for the catalog/image intelligence service.

Downloads Fashion-MNIST (a standard public apparel image dataset, fetched
directly from torchvision - not copied from any reference repository),
saves a sample of images as PNGs, and writes data/catalog.csv mapping each
sampled image to a `declared_category` (the catalog listing's stated
category) and the ground-truth `true_class`. Some declared categories are
deliberately set incorrectly so the demo has real category mismatches to
surface.
"""
import os
import random

import numpy as np
import pandas as pd
from torchvision import datasets
from torchvision.transforms.functional import to_pil_image

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Maps each Fashion-MNIST class to a Flipkart-style catalog category group.
CLASS_TO_GROUP = {
    "T-shirt/top": "Apparel",
    "Trouser": "Apparel",
    "Pullover": "Apparel",
    "Dress": "Apparel",
    "Coat": "Apparel",
    "Shirt": "Apparel",
    "Sandal": "Footwear",
    "Sneaker": "Footwear",
    "Ankle boot": "Footwear",
    "Bag": "Bags_Accessories",
}
CATALOG_GROUPS = sorted(set(CLASS_TO_GROUP.values()))

N_SAMPLES = 40
MISLABEL_RATE = 0.25
SEED = 7


def build(n_samples: int = N_SAMPLES, seed: int = SEED) -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(here, "raw_fashion_mnist")
    images_dir = os.path.join(here, "sample_images")
    os.makedirs(images_dir, exist_ok=True)

    test_set = datasets.FashionMNIST(root=raw_dir, train=False, download=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    indices = rng.sample(range(len(test_set)), n_samples)

    rows = []
    for i, idx in enumerate(indices):
        image, label = test_set[idx]
        true_class = FASHION_MNIST_CLASSES[label]
        true_group = CLASS_TO_GROUP[true_class]

        if np_rng.uniform() < MISLABEL_RATE:
            wrong_groups = [g for g in CATALOG_GROUPS if g != true_group]
            declared_group = np_rng.choice(wrong_groups)
        else:
            declared_group = true_group

        slug = true_class.lower().replace("/", "_").replace(" ", "_")
        filename = f"product_{i:03d}_{slug}.png"
        image.save(os.path.join(images_dir, filename))

        rows.append(
            {
                "product_id": f"P{i:04d}",
                "image_path": os.path.join("data", "sample_images", filename),
                "declared_category": declared_group,
                "true_class": true_class,
                "true_category": true_group,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    df = build()
    out_path = os.path.join(here, "catalog.csv")
    df.to_csv(out_path, index=False)
    n_mismatch = (df["declared_category"] != df["true_category"]).sum()
    print(f"Wrote {len(df)} catalog rows to {out_path}")
    print(f"Injected mismatches: {n_mismatch}/{len(df)}")
