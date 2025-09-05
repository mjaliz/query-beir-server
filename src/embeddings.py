import json
import logging
from typing import Any, Dict, Generator, List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(
            f"Loaded model {model_name} with embedding dimension {self.embedding_dim}"
        )

    def load_corpus(self, corpus_path: str) -> Generator[Dict[str, Any], None, None]:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line.strip())

    def load_qrels(self, qrels_path: str) -> Dict[str, List[str]]:
        corpus_to_queries = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    query_id, doc_id, score = parts[0], parts[1], float(parts[2])
                    if score > 0:  # Only store relevant pairs
                        if doc_id not in corpus_to_queries:
                            corpus_to_queries[doc_id] = []
                        corpus_to_queries[doc_id].append(query_id)
        logger.info(f"Loaded {len(corpus_to_queries)} corpus-query relevance pairs")
        return corpus_to_queries

    def embed_corpus_batch(
        self, corpus_batch: List[Dict[str, Any]]
    ) -> List[List[float]]:
        texts = [doc.get("title", "") for doc in corpus_batch]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_corpus(
        self, corpus_path: str, qrels_path: str
    ) -> Generator[Dict[str, Any], None, None]:
        corpus_to_queries = self.load_qrels(qrels_path)
        corpus_generator = self.load_corpus(corpus_path)

        batch = []
        total_processed = 0

        for doc in tqdm(corpus_generator, desc=f"Embedding with {self.model_name}"):
            batch.append(doc)

            if len(batch) >= self.batch_size:
                embeddings = self.embed_corpus_batch(batch)

                for i, (doc, embedding) in enumerate(zip(batch, embeddings)):
                    corpus_id = doc["_id"]
                    yield {
                        "id": corpus_id,
                        "vector": embedding,
                        "payload": {
                            "corpus_id": corpus_id,
                            "title": doc.get("title", ""),
                            "text": doc.get("text", ""),
                            "relevant_queries": corpus_to_queries.get(corpus_id, []),
                        },
                    }
                    total_processed += 1

                batch = []

        # Process remaining batch
        if batch:
            embeddings = self.embed_corpus_batch(batch)
            for doc, embedding in zip(batch, embeddings):
                corpus_id = doc["_id"]
                yield {
                    "id": corpus_id,
                    "vector": embedding,
                    "payload": {
                        "corpus_id": corpus_id,
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "relevant_queries": corpus_to_queries.get(corpus_id, []),
                    },
                }
                total_processed += 1

        logger.info(f"Embedded {total_processed} documents with {self.model_name}")

    def embed_queries(self, query_path: str) -> Dict[str, Dict[str, Any]]:
        queries = {}
        texts = []
        query_ids = []

        with open(query_path, "r", encoding="utf-8") as f:
            for line in f:
                query = json.loads(line.strip())
                query_ids.append(query["_id"])
                texts.append(query["text"])

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        for query_id, text, embedding in zip(query_ids, texts, embeddings):
            queries[query_id] = {
                "id": query_id,
                "text": text,
                "vector": embedding.tolist(),
            }

        logger.info(f"Embedded {len(queries)} queries with {self.model_name}")
        return queries
