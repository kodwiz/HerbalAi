"""Product catalogue loading and safe matching. Never invents products."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
from modules import safety_checker as sc
from modules.session_manager import UserProfile

REQUIRED_PRODUCT_FIELDS = ["id", "name", "category", "ingredients", "route", "herb_ids", "intended_use"]


@dataclass
class ProductCatalogue:
    products: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ProductCatalogue":
        try:
            raw = json.loads(Path(path or config.PRODUCTS_PATH).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls([])
        return cls(list(raw.get("products", [])))

    @classmethod
    def from_entries(cls, entries: list[dict[str, Any]]) -> "ProductCatalogue":
        return cls(list(entries))

    def get(self, product_id: str) -> dict[str, Any] | None:
        return next((p for p in self.products if p.get("id") == product_id), None)


@dataclass
class ProductMatch:
    product: dict[str, Any]
    safety: sc.SafetyResult


@dataclass
class ProductExclusion:
    product: dict[str, Any]
    reasons: list[str]


def match_products(catalogue: ProductCatalogue, herbs_by_id: dict[str, dict[str, Any]], profile: UserProfile,
                   candidate_herb_ids: set[str], routes: set[str]) -> tuple[list[ProductMatch], list[ProductExclusion]]:
    """Return products that pass every safety check, plus the ones that did not (with reasons)."""
    matches: list[ProductMatch] = []
    excluded: list[ProductExclusion] = []
    prefs = profile.preferences or {}
    for product in catalogue.products:
        missing = [f for f in REQUIRED_PRODUCT_FIELDS if not product.get(f)]
        if missing:
            excluded.append(ProductExclusion(product, [f"Incomplete product record (missing: {', '.join(missing)})."]))
            continue
        relevant = (set(product.get("herb_ids", [])) & candidate_herb_ids) or product.get("category") == profile.concern_category
        if not relevant or product.get("route") not in routes:
            continue
        reasons: list[str] = []
        if prefs.get("fragrance_free") and product.get("fragrance_free") is not True:
            reasons.append("You asked for fragrance-free products and this one is not confirmed fragrance-free.")
        safety = sc.check_product(product, herbs_by_id, profile)
        reasons += safety.blocks
        if reasons:
            excluded.append(ProductExclusion(product, reasons))
        else:
            matches.append(ProductMatch(product, safety))
    return matches, excluded
