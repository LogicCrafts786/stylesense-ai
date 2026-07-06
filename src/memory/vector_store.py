"""
Vector store wrapper supporting ChromaDB (default) or FAISS as a backend.

Used for two purposes:
1. Semantic search over the product catalog (e.g., "something breezy and
   elegant" matches products beyond simple keyword filters).
2. Long-term storage of user preference embeddings (used by
   user_profile_memory.py).

The backend is selected via VECTOR_STORE_TYPE in settings, and callers
interact with a single consistent interface regardless of backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.models.product import Product
from src.utils.config import get_settings
from src.utils.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    """Abstract interface that all vector store backends must implement."""

    @abstractmethod
    def index_products(self, products: list[Product]) -> None:
        """Embed and index a list of products for semantic search."""

    @abstractmethod
    def semantic_search(self, query: str, top_k: int = 10) -> list[str]:
        """Return product_ids most semantically similar to the query."""

    @abstractmethod
    def upsert_text(self, collection: str, doc_id: str, text: str, metadata: dict) -> None:
        """Insert or update an arbitrary text document (e.g., user preference notes)."""

    @abstractmethod
    def query_text(self, collection: str, query: str, top_k: int = 5) -> list[dict]:
        """Query an arbitrary text collection for semantically similar documents."""


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed implementation of the vector store interface."""

    def __init__(self) -> None:
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed.", details="Run: pip install chromadb"
            ) from exc

        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        self._product_collection = self._client.get_or_create_collection(
            name="products", embedding_function=self._embedding_fn
        )
        logger.info(f"ChromaVectorStore initialized at '{settings.chroma_persist_dir}'.")

    def index_products(self, products: list[Product]) -> None:
        """
        Embed and index products into the Chroma 'products' collection.

        Args:
            products: List of Product objects to index. Each product's
                searchable text combines name, description, style tags,
                and colors for richer semantic matching.
        """
        if not products:
            logger.warning("index_products called with an empty product list.")
            return

        documents = [self._product_to_search_text(p) for p in products]
        ids = [p.product_id for p in products]
        metadatas = [{"category": p.category.value, "price": p.price} for p in products]

        try:
            self._product_collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            logger.info(f"Indexed {len(products)} products into Chroma.")
        except Exception as exc:
            raise VectorStoreError("Failed to index products into Chroma.", details=str(exc)) from exc

    def semantic_search(self, query: str, top_k: int = 10) -> list[str]:
        """
        Perform a semantic search over indexed products.

        Args:
            query: Natural language search query.
            top_k: Maximum number of product IDs to return.

        Returns:
            List of matching product_ids, ranked by relevance.
        """
        try:
            results = self._product_collection.query(query_texts=[query], n_results=top_k)
            return results.get("ids", [[]])[0]
        except Exception as exc:
            logger.error(f"Chroma semantic search failed: {exc}")
            raise VectorStoreError("Semantic product search failed.", details=str(exc)) from exc

    def upsert_text(self, collection: str, doc_id: str, text: str, metadata: dict) -> None:
        """
        Insert or update a document in an arbitrary named collection
        (e.g., "user_preferences").

        Args:
            collection: Name of the Chroma collection to use.
            doc_id: Unique document ID (e.g., user_id).
            text: The text content to embed.
            metadata: Additional metadata to store alongside the embedding.
        """
        try:
            coll = self._client.get_or_create_collection(
                name=collection, embedding_function=self._embedding_fn
            )
            coll.upsert(documents=[text], ids=[doc_id], metadatas=[metadata])
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to upsert document '{doc_id}' into collection '{collection}'.",
                details=str(exc),
            ) from exc

    def query_text(self, collection: str, query: str, top_k: int = 5) -> list[dict]:
        """
        Query an arbitrary named collection for semantically similar documents.

        Args:
            collection: Name of the Chroma collection to query.
            query: Natural language query text.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: id, document, metadata.
        """
        try:
            coll = self._client.get_or_create_collection(
                name=collection, embedding_function=self._embedding_fn
            )
            results = coll.query(query_texts=[query], n_results=top_k)

            output = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            for doc_id, doc_text, meta in zip(ids, docs, metas):
                output.append({"id": doc_id, "document": doc_text, "metadata": meta})
            return output
        except Exception as exc:
            logger.error(f"Chroma query_text failed for collection '{collection}': {exc}")
            raise VectorStoreError("Text query failed.", details=str(exc)) from exc

    @staticmethod
    def _product_to_search_text(product: Product) -> str:
        """Build a rich text representation of a product for embedding."""
        return (
            f"{product.name}. {product.description}. "
            f"Category: {product.category.value}. "
            f"Colors: {', '.join(product.colors)}. "
            f"Style: {', '.join(product.style_tags)}. "
            f"Occasions: {', '.join(product.occasion_tags)}."
        )


class FaissVectorStore(BaseVectorStore):
    """
    FAISS-backed implementation of the vector store interface.

    Uses sentence-transformers style embeddings computed via a simple
    hashing-based fallback embedder if no embedding model is configured,
    to keep the project runnable without additional heavyweight
    dependencies. For production use, swap `_embed` to call a real
    embedding model.
    """

    def __init__(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise VectorStoreError(
                "faiss-cpu is not installed.", details="Run: pip install faiss-cpu"
            ) from exc

        self._faiss = faiss
        self._np = np
        self._dimension = 384
        self._index = faiss.IndexFlatL2(self._dimension)
        self._id_map: list[str] = []
        self._product_lookup: dict[str, Product] = {}
        self._text_collections: dict[str, dict[str, Any]] = {}

        settings = get_settings()
        logger.info(f"FaissVectorStore initialized (index path config: '{settings.faiss_index_path}').")

    def _embed(self, text: str):
        """
        Compute a deterministic pseudo-embedding for text.

        Note: This is a lightweight placeholder embedding (hash-based) to
        keep the project dependency-light and fully offline-runnable. For
        production semantic quality, replace with a real sentence embedding
        model (e.g., via langchain_google_genai embeddings).
        """
        import hashlib

        vector = self._np.zeros(self._dimension, dtype="float32")
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self._dimension
            vector[idx] += 1.0
        norm = self._np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def index_products(self, products: list[Product]) -> None:
        """Embed and index products into the FAISS flat index."""
        if not products:
            return

        vectors = self._np.array([self._embed(self._product_to_search_text(p)) for p in products])
        self._index.add(vectors)
        for p in products:
            self._id_map.append(p.product_id)
            self._product_lookup[p.product_id] = p
        logger.info(f"Indexed {len(products)} products into FAISS.")

    def semantic_search(self, query: str, top_k: int = 10) -> list[str]:
        """Search the FAISS index for the top_k most similar product IDs."""
        if self._index.ntotal == 0:
            return []

        query_vector = self._np.array([self._embed(query)])
        distances, indices = self._index.search(query_vector, min(top_k, self._index.ntotal))

        return [self._id_map[i] for i in indices[0] if 0 <= i < len(self._id_map)]

    def upsert_text(self, collection: str, doc_id: str, text: str, metadata: dict) -> None:
        """Store a text document in an in-memory dict-backed 'collection' (simplified for FAISS)."""
        self._text_collections.setdefault(collection, {})
        self._text_collections[collection][doc_id] = {
            "text": text,
            "vector": self._embed(text),
            "metadata": metadata,
        }

    def query_text(self, collection: str, query: str, top_k: int = 5) -> list[dict]:
        """Query an in-memory text collection using cosine similarity over stored vectors."""
        coll = self._text_collections.get(collection, {})
        if not coll:
            return []

        query_vector = self._embed(query)
        scored = []
        for doc_id, entry in coll.items():
            similarity = float(self._np.dot(query_vector, entry["vector"]))
            scored.append((similarity, doc_id, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": doc_id, "document": entry["text"], "metadata": entry["metadata"]}
            for _, doc_id, entry in scored[:top_k]
        ]

    @staticmethod
    def _product_to_search_text(product: Product) -> str:
        """Build a rich text representation of a product for embedding."""
        return (
            f"{product.name}. {product.description}. "
            f"Category: {product.category.value}. "
            f"Colors: {', '.join(product.colors)}. "
            f"Style: {', '.join(product.style_tags)}."
        )


def create_vector_store() -> BaseVectorStore:
    """
    Factory function that instantiates the configured vector store backend.

    Returns:
        A BaseVectorStore implementation (Chroma or FAISS) based on
        settings.vector_store_type.

    Raises:
        VectorStoreError: If the configured backend fails to initialize.
    """
    settings = get_settings()

    if settings.vector_store_type == "chroma":
        return ChromaVectorStore()
    elif settings.vector_store_type == "faiss":
        return FaissVectorStore()
    else:
        raise VectorStoreError(f"Unknown vector store type: '{settings.vector_store_type}'.")


_store_instance: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    """Return a lazily-initialized singleton vector store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = create_vector_store()
    return _store_instance
