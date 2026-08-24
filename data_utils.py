"""Small shared helpers to look up an order or catalog item by ID.
Used by both assistant.py and app.py so lookups aren't duplicated."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORDERS_PATH = os.path.join(HERE, "data", "orders.csv")
CATALOG_PATH = os.path.join(HERE, "data", "catalog.csv")

_orders = None
_catalog = None


def load_orders() -> pd.DataFrame:
    global _orders
    if _orders is None:
        _orders = pd.read_csv(ORDERS_PATH)
    return _orders


def load_catalog() -> pd.DataFrame:
    global _catalog
    if _catalog is None:
        _catalog = pd.read_csv(CATALOG_PATH)
    return _catalog


def get_order(order_id: int) -> dict | None:
    df = load_orders()
    row = df[df["order_id"] == order_id]
    if row.empty:
        return None
    record = row.iloc[0].to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in record.items()}


def get_catalog_item(product_id: str) -> dict | None:
    df = load_catalog()
    row = df[df["product_id"] == product_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()
