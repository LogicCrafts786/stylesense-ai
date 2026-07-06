"""Unit tests for src.services.product_catalog_service.ProductCatalogService."""

from __future__ import annotations

import json

import pytest

from src.models.product import ProductCategory
from src.services.product_catalog_service import ProductCatalogService
from src.utils.exceptions import ProductCatalogError

_SAMPLE_CATALOG = [
    {
        "product_id": "P1",
        "name": "Test Top",
        "category": "top",
        "price": 40.0,
        "colors": ["navy"],
        "style_tags": ["classic"],
        "occasion_tags": ["office"],
        "in_stock": True,
    },
    {
        "product_id": "P2",
        "name": "Expensive Shoes",
        "category": "shoes",
        "price": 300.0,
        "colors": ["black"],
        "style_tags": ["elegant"],
        "occasion_tags": ["gala"],
        "in_stock": True,
    },
    {
        "product_id": "P3",
        "name": "Out of Stock Bottom",
        "category": "bottom",
        "price": 30.0,
        "colors": ["khaki"],
        "style_tags": ["casual"],
        "occasion_tags": ["casual outing"],
        "in_stock": False,
    },
]


@pytest.fixture
def catalog_service(tmp_path) -> ProductCatalogService:
    catalog_file = tmp_path / "sample_products.json"
    catalog_file.write_text(json.dumps(_SAMPLE_CATALOG), encoding="utf-8")
    return ProductCatalogService(catalog_path=catalog_file)


class TestProductCatalogService:
    def test_loads_all_products(self, catalog_service):
        assert len(catalog_service.get_all_products()) == 3

    def test_get_by_id_found(self, catalog_service):
        product = catalog_service.get_by_id("P1")
        assert product is not None
        assert product.name == "Test Top"

    def test_get_by_id_not_found(self, catalog_service):
        assert catalog_service.get_by_id("NONEXISTENT") is None

    def test_filter_by_category(self, catalog_service):
        results = catalog_service.filter_products(category=ProductCategory.SHOES)
        assert len(results) == 1
        assert results[0].product_id == "P2"

    def test_filter_by_max_price(self, catalog_service):
        results = catalog_service.filter_products(max_price=50.0)
        product_ids = {p.product_id for p in results}
        assert product_ids == {"P1"}  # P3 excluded (out of stock), P2 excluded (too expensive)

    def test_in_stock_only_excludes_out_of_stock(self, catalog_service):
        results = catalog_service.filter_products(in_stock_only=True)
        assert all(p.product_id != "P3" for p in results)

    def test_in_stock_false_includes_all(self, catalog_service):
        results = catalog_service.filter_products(in_stock_only=False)
        product_ids = {p.product_id for p in results}
        assert "P3" in product_ids

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ProductCatalogError):
            ProductCatalogService(catalog_path=tmp_path / "does_not_exist.json")

    def test_malformed_file_raises(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ProductCatalogError):
            ProductCatalogService(catalog_path=bad_file)
