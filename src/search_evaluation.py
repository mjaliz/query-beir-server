import json
from typing import List, Dict, Any, Set, Tuple
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    query_id: str
    query_text: str
    retrieved_docs: List[str]
    relevant_docs: List[str]
    scores: List[float]
    payloads: List[Dict[str, Any]]
    false_positives: List[str]
    true_positives: List[str]
    false_negatives: List[str]
    precision: float
    recall: float
    f1: float

class SearchEvaluator:
    def __init__(self, qdrant_storage, collection_name: str):
        self.qdrant_storage = qdrant_storage
        self.collection_name = collection_name
        self.qrels = {}
        self.results = []

    def load_qrels(self, qrels_path: str):
        with open(qrels_path, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    query_id, doc_id, score = parts[0], parts[1], float(parts[2])
                    if score > 0:  # Only store relevant pairs
                        if query_id not in self.qrels:
                            self.qrels[query_id] = set()
                        self.qrels[query_id].add(doc_id)
        logger.info(f"Loaded relevance judgments for {len(self.qrels)} queries")

    def search_queries(self, queries: Dict[str, Dict[str, Any]], top_k: int = 10) -> List[SearchResult]:
        results = []
        
        for query_id, query_data in tqdm(queries.items(), desc="Searching queries"):
            # Search in Qdrant
            search_results = self.qdrant_storage.search(
                collection_name=self.collection_name,
                query_vector=query_data['vector'],
                top_k=top_k
            )
            
            # Extract corpus IDs from payloads (not the Qdrant UUID), scores, and payloads
            retrieved_docs = [point.payload.get('corpus_id', point.id) for point in search_results]
            scores = [point.score for point in search_results]
            payloads = [point.payload for point in search_results]
            
            # Get relevant documents for this query
            relevant_docs = list(self.qrels.get(query_id, set()))
            
            # Calculate metrics
            result = self._evaluate_single_query(
                query_id=query_id,
                query_text=query_data['text'],
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
                scores=scores,
                payloads=payloads
            )
            
            results.append(result)
        
        self.results = results
        return results

    def _evaluate_single_query(self, query_id: str, query_text: str,
                              retrieved_docs: List[str], relevant_docs: List[str],
                              scores: List[float], payloads: List[Dict[str, Any]]) -> SearchResult:
        retrieved_set = set(retrieved_docs)
        relevant_set = set(relevant_docs)
        
        # Calculate true positives, false positives, false negatives
        true_positives = list(retrieved_set & relevant_set)
        false_positives = list(retrieved_set - relevant_set)
        false_negatives = list(relevant_set - retrieved_set)
        
        # Calculate metrics
        precision = len(true_positives) / len(retrieved_docs) if retrieved_docs else 0.0
        recall = len(true_positives) / len(relevant_docs) if relevant_docs else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return SearchResult(
            query_id=query_id,
            query_text=query_text,
            retrieved_docs=retrieved_docs,
            relevant_docs=relevant_docs,
            scores=scores,
            payloads=payloads,
            false_positives=false_positives,
            true_positives=true_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1
        )

    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        if not self.results:
            return {}
        
        precisions = [r.precision for r in self.results]
        recalls = [r.recall for r in self.results]
        f1s = [r.f1 for r in self.results]
        
        return {
            "avg_precision": np.mean(precisions),
            "avg_recall": np.mean(recalls),
            "avg_f1": np.mean(f1s),
            "std_precision": np.std(precisions),
            "std_recall": np.std(recalls),
            "std_f1": np.std(f1s),
            "total_queries": len(self.results),
            "queries_with_relevant_docs": sum(1 for r in self.results if r.relevant_docs),
            "queries_with_retrieved_docs": sum(1 for r in self.results if r.retrieved_docs)
        }

    def save_false_positives(self, output_path: str):
        false_positive_records = []
        
        for result in self.results:
            if result.false_positives:
                # Get details of false positive documents from stored payloads
                for fp_doc_id in result.false_positives:
                    # Find the index and get score and payload for this document
                    doc_index = result.retrieved_docs.index(fp_doc_id)
                    score = result.scores[doc_index] if doc_index < len(result.scores) else 0.0
                    payload = result.payloads[doc_index] if doc_index < len(result.payloads) else {}
                    
                    # Extract title and corpus_id from payload
                    false_positive_title = payload.get("title", "")
                    corpus_id = payload.get("corpus_id", fp_doc_id)
                    
                    # Extract queries information from payload
                    # For false positives, we want to show which queries this document IS relevant for
                    # (to understand why it might have been incorrectly retrieved)
                    relevant_queries = payload.get("relevant_queries", [])
                    fp_relevant_query_ids = []
                    fp_relevant_query_texts = []
                    for query_info in relevant_queries:
                        if isinstance(query_info, dict):
                            fp_relevant_query_ids.append(query_info.get("query_id", ""))
                            fp_relevant_query_texts.append(query_info.get("query_text", ""))
                        else:
                            # Handle old format where it might just be query_id strings
                            fp_relevant_query_ids.append(query_info)
                            fp_relevant_query_texts.append("")
                    
                    # Also get non-relevant queries if needed
                    non_relevant_queries = payload.get("non_relevant_queries", [])
                    fp_non_relevant_query_ids = []
                    fp_non_relevant_query_texts = []
                    for query_info in non_relevant_queries:
                        if isinstance(query_info, dict):
                            fp_non_relevant_query_ids.append(query_info.get("query_id", ""))
                            fp_non_relevant_query_texts.append(query_info.get("query_text", ""))
                        else:
                            fp_non_relevant_query_ids.append(query_info)
                            fp_non_relevant_query_texts.append("")
                    
                    record = {
                        "query_id": result.query_id,
                        "query_text": result.query_text,
                        "false_positive_title": false_positive_title,
                        "corpus_id": corpus_id,
                        "fp_relevant_query_ids": fp_relevant_query_ids,
                        "fp_relevant_query_texts": fp_relevant_query_texts,
                        "fp_non_relevant_query_ids": fp_non_relevant_query_ids,
                        "fp_non_relevant_query_texts": fp_non_relevant_query_texts,
                        "score": score,
                    }
                    false_positive_records.append(record)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in false_positive_records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(false_positive_records)} false positive records to {output_path}")

    def print_summary(self):
        metrics = self.calculate_aggregate_metrics()
        
        print("\n" + "="*50)
        print("SEARCH EVALUATION SUMMARY")
        print("="*50)
        print(f"Collection: {self.collection_name}")
        print(f"Total Queries: {metrics.get('total_queries', 0)}")
        print(f"Queries with Relevant Docs: {metrics.get('queries_with_relevant_docs', 0)}")
        print(f"Queries with Retrieved Docs: {metrics.get('queries_with_retrieved_docs', 0)}")
        print("-"*50)
        print(f"Average Precision: {metrics.get('avg_precision', 0):.4f} (±{metrics.get('std_precision', 0):.4f})")
        print(f"Average Recall: {metrics.get('avg_recall', 0):.4f} (±{metrics.get('std_recall', 0):.4f})")
        print(f"Average F1: {metrics.get('avg_f1', 0):.4f} (±{metrics.get('std_f1', 0):.4f})")
        print("="*50)