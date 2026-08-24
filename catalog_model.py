"""
Part 2 - Catalog / Image Intelligence
======================================
Classifies a product image and checks it against the catalog's declared
category, flagging mismatches or low-confidence (likely mislabeled) listings.

Run this file directly to train:
    python catalog_model.py

Import it elsewhere to get predictions once trained:
    from catalog_model import predict, check_category

Data note: real Flipkart product images are not available, so this uses
Fashion-MNIST (a public apparel image dataset) as a stand-in sample dataset
via data/build_catalog.py. This is clearly NOT real Flipkart catalog data.
"""
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "data", "raw_fashion_mnist")
MODEL_DIR = os.path.join(HERE, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "catalog_classifier.pt")
META_PATH = os.path.join(MODEL_DIR, "catalog_meta.json")

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
CLASS_TO_GROUP = {
    "T-shirt/top": "Apparel", "Trouser": "Apparel", "Pullover": "Apparel",
    "Dress": "Apparel", "Coat": "Apparel", "Shirt": "Apparel",
    "Sandal": "Footwear", "Sneaker": "Footwear", "Ankle boot": "Footwear",
    "Bag": "Bags_Accessories",
}

IMAGE_SIZE = 112
LOW_CONFIDENCE_THRESHOLD = 0.55
# Small subset + frozen backbone keeps training under a couple of minutes on CPU.
TRAIN_SUBSET_SIZE = 3000
TEST_SUBSET_SIZE = 1000
EPOCHS = 2
BATCH_SIZE = 64

_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _build_model() -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))  # only this layer trains
    return model


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device("cpu")

    train_full = datasets.FashionMNIST(root=RAW_DIR, train=True, download=True, transform=_transform)
    test_full = datasets.FashionMNIST(root=RAW_DIR, train=False, download=True, transform=_transform)
    train_set = Subset(train_full, list(range(TRAIN_SUBSET_SIZE)))
    test_set = Subset(test_full, list(range(TEST_SUBSET_SIZE)))

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    model = _build_model().to(device)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        print(f"Epoch {epoch + 1}/{EPOCHS} - avg loss {total_loss / len(train_set):.4f}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    print(f"Test accuracy on {total} sample images: {accuracy:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    with open(META_PATH, "w") as f:
        json.dump({"classes": CLASSES, "class_to_group": CLASS_TO_GROUP, "test_accuracy": round(accuracy, 4)}, f, indent=2)
    print(f"Saved model to {MODEL_PATH}")


# ---------------------------------------------------------------------------
# Prediction (used by the assistant and the Streamlit app)
# ---------------------------------------------------------------------------
_model = None
_meta = None


def _load():
    global _model, _meta
    if _model is None:
        with open(META_PATH) as f:
            _meta = json.load(f)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(_meta["classes"]))
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _model = m
    return _model, _meta


def predict(image_path: str) -> dict:
    model, meta = _load()
    abs_path = image_path if os.path.isabs(image_path) else os.path.join(HERE, image_path)
    image = Image.open(abs_path).convert("L")
    tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    idx = int(torch.argmax(probs).item())
    predicted_class = meta["classes"][idx]
    return {
        "predicted_class": predicted_class,
        "predicted_category": meta["class_to_group"][predicted_class],
        "confidence": round(float(probs[idx]), 4),
    }


def check_category(image_path: str, declared_category: str | None = None) -> dict:
    result = predict(image_path)
    issue = None
    match = None
    if result["confidence"] < LOW_CONFIDENCE_THRESHOLD:
        issue = f"Low-confidence prediction ({result['confidence']:.0%}) — image may be ambiguous or low quality"
    if declared_category:
        match = declared_category == result["predicted_category"]
        if not match:
            note = (
                f"Declared category '{declared_category}' does not match predicted "
                f"category '{result['predicted_category']}' (looks like {result['predicted_class']})"
            )
            issue = f"{issue}; {note}" if issue else note
    return {**result, "declared_category": declared_category, "category_match": match, "issue": issue}


if __name__ == "__main__":
    train_and_evaluate()
