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
            
            # Extract retrieved document IDs
            retrieved_docs = [point.id for point in search_results]
            scores = [point.score for point in search_results]
            
            # Get relevant documents for this query
            relevant_docs = list(self.qrels.get(query_id, set()))
            
            # Calculate metrics
            result = self._evaluate_single_query(
                query_id=query_id,
                query_text=query_data['text'],
                retrieved_docs=retrieved_docs,
                relevant_docs=relevant_docs,
                scores=scores
            )
            
            results.append(result)
        
        self.results = results
        return results

    def _evaluate_single_query(self, query_id: str, query_text: str,
                              retrieved_docs: List[str], relevant_docs: List[str],
                              scores: List[float]) -> SearchResult:
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
                # Get details of false positive documents from Qdrant
                for fp_doc_id in result.false_positives:
                    # Find the score for this document
                    doc_index = result.retrieved_docs.index(fp_doc_id)
                    score = result.scores[doc_index] if doc_index < len(result.scores) else 0.0
                    
                    # Retrieve the document from Qdrant to get its title
                    try:
                        doc_points = self.qdrant_storage.client.retrieve(
                            collection_name=self.collection_name,
                            ids=[fp_doc_id]
                        )
                        false_positive_title = ""
                        if doc_points:
                            false_positive_title = doc_points[0].payload.get("title", "")
                    except Exception as e:
                        logger.warning(f"Could not retrieve title for doc {fp_doc_id}: {e}")
                        false_positive_title = ""
                    
                    record = {
                        "query_id": result.query_id,
                        "query_text": result.query_text,
                        "false_positive_title": false_positive_title,
                        "score": score,
                        "relevant_docs": result.relevant_docs,
                        "rank": doc_index + 1
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