from dataclasses import dataclass
import re

from sqlalchemy.orm import Session

from app.core.schemas import RetrievedChunk
from app.db import crud
from app.db.models import Chunk
from app.retrieval.bm25_store import BM25Store
from app.retrieval.embedder import Embedder
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import FaissVectorStore


SUMMARY_INTENT_WORDS = ("总结", "梳理", "归纳", "复习", "提纲")
SUMMARY_EXPANSION_SUFFIXES = ("定义", "主要方法", "分类", "优缺点", "应用")


@dataclass(frozen=True)
class RetrievalProfileConfig:
    vector_top_k: int
    bm25_top_k: int
    vector_weight: float
    bm25_weight: float


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: FaissVectorStore,
        bm25_store: BM25Store,
        reranker: Reranker | None = None,
        vector_top_k: int = 10,
        bm25_top_k: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        enable_reranker: bool = True,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.reranker = reranker
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.enable_reranker = enable_reranker

    def rebuild(self, db: Session) -> None:
        chunks = crud.list_chunks(db)
        if not chunks:
            self.vector_store.build([], None)
            self.bm25_store.build([], [])
            return
        embeddings = self.embedder.encode_texts([chunk.text for chunk in chunks])
        self.vector_store.build([chunk.id for chunk in chunks], embeddings)
        self.bm25_store.build([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])

    def retrieve(self, db: Session, query: str, top_k: int, profile: str = "qa") -> list[RetrievedChunk]:
        self._ensure_indexes(db)
        normalized_profile = _normalize_profile(profile)
        if normalized_profile == "summary":
            return self._retrieve_summary(db, query, top_k)

        config = self._profile_config(normalized_profile)
        ranked = self._retrieve_single_query(db, query, config)
        if self.enable_reranker and self.reranker is not None:
            return self.reranker.rerank(query, ranked, top_k)
        return ranked[:top_k]

    def _retrieve_summary(self, db: Session, query: str, top_k: int) -> list[RetrievedChunk]:
        config = self._profile_config("summary")
        merged: dict[str, RetrievedChunk] = {}
        for expanded_query in expand_summary_queries(query):
            candidates = self._retrieve_single_query(db, expanded_query, config)
            for candidate in candidates:
                current = merged.get(candidate.chunk_id)
                if current is None or candidate.final_score > current.final_score:
                    merged[candidate.chunk_id] = candidate

        ranked = sorted(merged.values(), key=lambda item: item.final_score, reverse=True)
        rerank_limit = max(top_k * 3, top_k)
        if self.enable_reranker and self.reranker is not None:
            ranked = self.reranker.rerank(query, ranked, rerank_limit)
        else:
            ranked = ranked[:rerank_limit]
        return _select_diverse_chunks(ranked, top_k)

    def _retrieve_single_query(
        self,
        db: Session,
        query: str,
        config: RetrievalProfileConfig,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.encode_query(query)
        vector_results = self.vector_store.search(query_embedding, config.vector_top_k)
        bm25_results = self.bm25_store.search(query, config.bm25_top_k)
        merged_scores = self._merge_scores(vector_results, bm25_results, config)
        chunk_ids = list(merged_scores.keys())
        chunks = crud.get_chunks_by_ids(db, chunk_ids)
        by_id = {chunk.id: chunk for chunk in chunks}

        results: list[RetrievedChunk] = []
        for chunk_id, scores in merged_scores.items():
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(self._to_retrieved_chunk(chunk, scores))

        return sorted(results, key=lambda item: item.final_score, reverse=True)

    def _ensure_indexes(self, db: Session) -> None:
        chunks = crud.list_chunks(db)
        if not chunks:
            return
        if not self.bm25_store.path.exists():
            self.bm25_store.build([chunk.id for chunk in chunks], [chunk.text for chunk in chunks])

    def _merge_scores(
        self,
        vector_results: list[tuple[str, float]],
        bm25_results: list[tuple[str, float]],
        config: RetrievalProfileConfig,
    ) -> dict[str, dict[str, float | None]]:
        vector_norm = _normalize_scores(vector_results)
        bm25_norm = _normalize_scores(bm25_results)
        chunk_ids = list(dict.fromkeys([chunk_id for chunk_id, _ in vector_results + bm25_results]))

        merged: dict[str, dict[str, float | None]] = {}
        for chunk_id in chunk_ids:
            vector_score = next((score for cid, score in vector_results if cid == chunk_id), None)
            bm25_score = next((score for cid, score in bm25_results if cid == chunk_id), None)
            final_score = (
                config.vector_weight * vector_norm.get(chunk_id, 0.0)
                + config.bm25_weight * bm25_norm.get(chunk_id, 0.0)
            )
            merged[chunk_id] = {
                "vector_score": vector_score,
                "bm25_score": bm25_score,
                "final_score": final_score,
            }
        return merged

    def _to_retrieved_chunk(self, chunk: Chunk, scores: dict[str, float | None]) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            text=chunk.text,
            source_file=chunk.source_file,
            page=chunk.page,
            heading=chunk.heading,
            vector_score=scores["vector_score"],
            bm25_score=scores["bm25_score"],
            final_score=scores["final_score"] or 0.0,
        )

    def _profile_config(self, profile: str) -> RetrievalProfileConfig:
        if profile == "qa":
            return RetrievalProfileConfig(
                vector_top_k=12,
                bm25_top_k=12,
                vector_weight=0.65,
                bm25_weight=0.35,
            )
        if profile == "summary":
            return RetrievalProfileConfig(
                vector_top_k=20,
                bm25_top_k=20,
                vector_weight=0.6,
                bm25_weight=0.4,
            )
        return RetrievalProfileConfig(
            vector_top_k=self.vector_top_k,
            bm25_top_k=self.bm25_top_k,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )


def _normalize_scores(results: list[tuple[str, float]]) -> dict[str, float]:
    if not results:
        return {}
    values = [score for _, score in results]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {chunk_id: 1.0 for chunk_id, _ in results}
    return {chunk_id: (score - min_score) / (max_score - min_score) for chunk_id, score in results}


def _normalize_profile(profile: str) -> str:
    profile = (profile or "qa").strip().lower()
    if profile in {"qa", "summary", "default"}:
        return profile
    return "default"


def expand_summary_queries(query: str) -> list[str]:
    topic = query.strip()
    for word in SUMMARY_INTENT_WORDS:
        topic = topic.replace(word, "")
    topic = re.sub(r"一下|一?个|的|主要|有哪些|是什么|？|\?", "", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ，,。；;：:")
    if not topic:
        return [query]

    queries = [query]
    queries.extend(f"{topic} {suffix}" for suffix in SUMMARY_EXPANSION_SUFFIXES)
    return list(dict.fromkeys(queries))


def _select_diverse_chunks(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    selected: list[RetrievedChunk] = []
    seen_groups: set[tuple[str, str]] = set()

    for chunk in chunks:
        group = _diversity_group(chunk)
        if group in seen_groups:
            continue
        selected.append(chunk)
        seen_groups.add(group)
        if len(selected) >= top_k:
            return selected

    selected_ids = {chunk.chunk_id for chunk in selected}
    for chunk in chunks:
        if chunk.chunk_id in selected_ids:
            continue
        selected.append(chunk)
        if len(selected) >= top_k:
            break
    return selected


def _diversity_group(chunk: RetrievedChunk) -> tuple[str, str]:
    if chunk.heading:
        return (chunk.document_id, f"heading:{chunk.heading}")
    if chunk.page is not None:
        return (chunk.document_id, f"page:{chunk.page}")
    return (chunk.document_id, f"chunk:{chunk.chunk_id}")


def estimate_confidence(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "low"
    top_score = max(chunk.final_score for chunk in chunks)
    if top_score >= 0.75:
        return "high"
    if top_score >= 0.45:
        return "medium"
    return "low"
