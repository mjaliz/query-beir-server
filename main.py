import logging
import sys
from pathlib import Path
from typing import Optional
import argparse

from src.config import config
from src.embeddings import EmbeddingService
from src.qdrant_storage import QdrantStorage
from src.search_evaluation import SearchEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def embed_and_index(model_name: str, qdrant_storage: QdrantStorage, 
                   corpus_path: str, qrels_path: str, query_path: str):
    """Embed corpus and index in Qdrant"""
    logger.info(f"Starting embedding and indexing with model: {model_name}")
    
    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=model_name,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size
    )
    
    # Get collection name
    collection_name = qdrant_storage.get_collection_name(model_name)
    
    # Create collection
    qdrant_storage.create_collection(
        collection_name=collection_name,
        vector_size=embedding_service.embedding_dim
    )
    
    # Embed and index corpus
    embedding_generator = embedding_service.embed_corpus(corpus_path, qrels_path)
    qdrant_storage.index_corpus(collection_name, embedding_generator)
    
    # Get collection info
    info = qdrant_storage.get_collection_info(collection_name)
    logger.info(f"Collection info: {info}")
    
    return collection_name, embedding_service

def search_and_evaluate(model_name: str, collection_name: str, 
                        embedding_service: EmbeddingService,
                        qdrant_storage: QdrantStorage):
    """Search queries and evaluate results"""
    logger.info(f"Starting search and evaluation for collection: {collection_name}")
    
    # Initialize evaluator
    evaluator = SearchEvaluator(qdrant_storage, collection_name)
    
    # Load relevance judgments
    evaluator.load_qrels(config.data.qrels_path)
    
    # Embed queries
    queries = embedding_service.embed_queries(config.data.query_path)
    
    # Search and evaluate
    results = evaluator.search_queries(queries, top_k=config.search.top_k)
    
    # Save false positives
    false_positives_path = f"{config.data.false_positives_output}.{model_name.replace('/', '_')}.jsonl"
    evaluator.save_false_positives(false_positives_path)
    
    # Print summary
    evaluator.print_summary()
    
    return evaluator.calculate_aggregate_metrics()

def main():
    parser = argparse.ArgumentParser(description='BEIR Corpus Embedding and Search Evaluation')
    parser.add_argument('--models', nargs='+', help='List of embedding models to use')
    parser.add_argument('--embed-only', action='store_true', help='Only embed and index, skip search')
    parser.add_argument('--search-only', action='store_true', help='Only search, skip embedding')
    parser.add_argument('--collection', type=str, help='Collection name for search-only mode')
    
    args = parser.parse_args()
    
    # Use models from args or config
    models = args.models if args.models else config.embedding.models
    
    # Initialize Qdrant storage
    qdrant_storage = QdrantStorage(
        host=config.qdrant.host,
        port=config.qdrant.port,
        api_key=config.qdrant.api_key,
        https=config.qdrant.https
    )
    
    all_metrics = {}
    
    for model_name in models:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*50}")
            
            if not args.search_only:
                # Embed and index
                collection_name, embedding_service = embed_and_index(
                    model_name=model_name,
                    qdrant_storage=qdrant_storage,
                    corpus_path=config.data.corpus_path,
                    qrels_path=config.data.qrels_path,
                    query_path=config.data.query_path
                )
            else:
                # Use existing collection
                collection_name = args.collection or qdrant_storage.get_collection_name(model_name)
                embedding_service = EmbeddingService(
                    model_name=model_name,
                    device=config.embedding.device,
                    batch_size=config.embedding.batch_size
                )
            
            if not args.embed_only:
                # Search and evaluate
                metrics = search_and_evaluate(
                    model_name=model_name,
                    collection_name=collection_name,
                    embedding_service=embedding_service,
                    qdrant_storage=qdrant_storage
                )
                all_metrics[model_name] = metrics
            
        except Exception as e:
            logger.error(f"Error processing model {model_name}: {e}")
            continue
    
    # Print comparison if multiple models
    if len(all_metrics) > 1 and not args.embed_only:
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        for model_name, metrics in all_metrics.items():
            print(f"\n{model_name}:")
            print(f"  Precision: {metrics['avg_precision']:.4f}")
            print(f"  Recall: {metrics['avg_recall']:.4f}")
            print(f"  F1: {metrics['avg_f1']:.4f}")
        print("="*50)

if __name__ == "__main__":
    main()