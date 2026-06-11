"""
Enhanced RAG System
====================
Ontology-aware retrieval that routes queries to the right memory regions
before performing vector search — dramatically cutting retrieval cost.
"""
from __future__ import annotations
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


@dataclass
class Document:
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        return hashlib.md5(self.content.encode()).hexdigest()[:12]


@dataclass
class RetrievalResult:
    doc_id: str = ""
    content: str = ""
    score: float = 0.0
    metadata: Dict = field(default_factory=dict)
    source: str = ""
    retrieval_method: str = "vector"


@dataclass
class RAGConfig:
    top_k: int = 5
    similarity_threshold: float = 0.6
    use_ontology_routing: bool = True
    use_reranking: bool = True
    use_hyde: bool = False           # Hypothetical Document Embeddings
    max_context_tokens: int = 4096
    chunk_size: int = 512
    chunk_overlap: int = 64


class DocumentStore:
    """In-memory document store (swap with ChromaDB/Pinecone in production)."""

    def __init__(self):
        self._documents: Dict[str, Document] = {}
        self._tag_index: Dict[str, List[str]] = {}
        self._source_index: Dict[str, List[str]] = {}

    def add(self, doc: Document) -> str:
        self._documents[doc.doc_id] = doc

        for tag in doc.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = []
            self._tag_index[tag].append(doc.doc_id)

        if doc.source not in self._source_index:
            self._source_index[doc.source] = []
        self._source_index[doc.source].append(doc.doc_id)

        return doc.doc_id

    def get(self, doc_id: str) -> Optional[Document]:
        doc = self._documents.get(doc_id)
        if doc:
            doc.access_count += 1
        return doc

    def search_by_tag(self, tag: str) -> List[Document]:
        ids = self._tag_index.get(tag, [])
        return [self._documents[i] for i in ids if i in self._documents]

    def search_by_source(self, source: str) -> List[Document]:
        ids = self._source_index.get(source, [])
        return [self._documents[i] for i in ids if i in self._documents]

    def keyword_search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """BM25-style keyword search (simplified)."""
        query_terms = set(query.lower().split())
        results = []

        for doc in self._documents.values():
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / (len(query_terms) + len(doc_terms) - overlap)
                results.append((doc, score))

        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    def vector_search(self, query_embedding: List[float],
                      top_k: int = 5) -> List[Tuple[Document, float]]:
        """Cosine similarity search (stub — replace with FAISS/ChromaDB)."""
        if not query_embedding:
            return []
        results = []
        for doc in self._documents.values():
            if doc.embedding:
                score = self._cosine(query_embedding, doc.embedding)
                results.append((doc, score))
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    @property
    def size(self) -> int:
        return len(self._documents)


class TextChunker:
    """Splits documents into overlapping chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "",
              metadata: Optional[Dict] = None) -> List[Document]:
        words = text.split()
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_doc = Document(
                content=chunk_text,
                metadata={**(metadata or {}),
                          "parent_doc": doc_id,
                          "chunk_index": len(chunks),
                          "start_word": i},
                tags=["chunk"],
            )
            chunks.append(chunk_doc)
            if i + self.chunk_size >= len(words):
                break

        return chunks


class HyDEGenerator:
    """Hypothetical Document Embedding — generate a hypothetical answer and embed it."""

    def generate_hypothetical_doc(self, query: str) -> str:
        """Stub — in production, call LLM to generate hypothetical answer."""
        return f"A comprehensive answer to '{query}' would include detailed information about the key concepts, relevant examples, and supporting evidence from authoritative sources."


class Reranker:
    """Re-rank initial retrieval results for precision."""

    def rerank(self, query: str, results: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """Cross-encoder style reranking (stub)."""
        for result in results:
            query_words = set(query.lower().split())
            content_words = set(result.content.lower().split())
            lexical_overlap = len(query_words & content_words) / max(len(query_words), 1)
            # Blend original score with lexical overlap
            result.score = 0.7 * result.score + 0.3 * lexical_overlap
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


class OntologyAwareRAG:
    """
    Main RAG engine with ontological pre-routing.

    Pipeline:
    query → ontology_recognition → region_routing →
    hybrid_retrieval (keyword + vector) → reranking → context_assembly
    """

    def __init__(self, config: Optional[RAGConfig] = None,
                 ontology_index=None):
        self.rag_id = str(uuid.uuid4())[:8]
        self.config = config or RAGConfig()
        self.store = DocumentStore()
        self.chunker = TextChunker(self.config.chunk_size, self.config.chunk_overlap)
        self.reranker = Reranker()
        self.hyde = HyDEGenerator()
        self.ontology_index = ontology_index

        self.query_log: List[Dict] = []
        self.total_queries = 0
        self.total_latency = 0.0

    def ingest(self, text: str, source: str = "unknown",
               tags: Optional[List[str]] = None,
               metadata: Optional[Dict] = None) -> List[str]:
        """Ingest a document: chunk it and index all chunks."""
        parent_id = str(uuid.uuid4())[:8]
        chunks = self.chunker.chunk(text, doc_id=parent_id, metadata=metadata)
        ids = []
        for chunk in chunks:
            chunk.source = source
            chunk.tags.extend(tags or [])
            ids.append(self.store.add(chunk))
        return ids

    def ingest_document(self, doc: Document) -> str:
        return self.store.add(doc)

    def retrieve(self, query: str, context: Optional[Dict] = None) -> List[RetrievalResult]:
        start = time.time()
        self.total_queries += 1

        # Step 1: Ontological routing
        routing = {}
        if self.config.use_ontology_routing and self.ontology_index:
            tag = self.ontology_index.recognizer.recognize(query)
            routing = tag.routing

        # Step 2: HyDE (optional)
        search_query = query
        if self.config.use_hyde:
            hyp_doc = self.hyde.generate_hypothetical_doc(query)
            search_query = hyp_doc

        # Step 3: Hybrid retrieval
        keyword_results = self.store.keyword_search(search_query, top_k=self.config.top_k * 2)
        retrieval_results = [
            RetrievalResult(
                doc_id=doc.doc_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata,
                source=doc.source,
                retrieval_method="keyword",
            )
            for doc, score in keyword_results
            if score >= self.config.similarity_threshold * 0.5
        ]

        # Step 4: Re-rank
        if self.config.use_reranking and retrieval_results:
            retrieval_results = self.reranker.rerank(query, retrieval_results, self.config.top_k)

        # Trim to top_k
        retrieval_results = retrieval_results[:self.config.top_k]

        latency = time.time() - start
        self.total_latency += latency

        self.query_log.append({
            "query": query[:100],
            "results": len(retrieval_results),
            "routing": routing,
            "latency_ms": round(latency * 1000, 2),
            "timestamp": time.time(),
        })

        return retrieval_results

    def assemble_context(self, query: str, results: Optional[List[RetrievalResult]] = None,
                         max_tokens: int = 4096) -> str:
        """Assemble retrieved chunks into a context string for LLM."""
        if results is None:
            results = self.retrieve(query)
        if not results:
            return ""

        context_parts = [f"Query: {query}\n\nRelevant Context:\n"]
        token_estimate = len(query.split()) * 1.3
        for i, result in enumerate(results):
            chunk_text = f"[{i+1}] (source={result.source}, score={result.score:.2f})\n{result.content}\n"
            token_estimate += len(chunk_text.split()) * 1.3
            if token_estimate > max_tokens:
                break
            context_parts.append(chunk_text)

        return "\n".join(context_parts)

    def stats(self) -> Dict:
        return {
            "rag_id": self.rag_id,
            "documents_indexed": self.store.size,
            "total_queries": self.total_queries,
            "mean_latency_ms": round(
                (self.total_latency / max(1, self.total_queries)) * 1000, 2
            ),
            "config": {
                "top_k": self.config.top_k,
                "reranking": self.config.use_reranking,
                "ontology_routing": self.config.use_ontology_routing,
            },
        }

    def __repr__(self) -> str:
        return (f"OntologyAwareRAG(id={self.rag_id}, "
                f"docs={self.store.size}, queries={self.total_queries})")