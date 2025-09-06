import logging
import sys
from pathlib import Path
from typing import Optional
import argparse

from src.config import config
from src.embeddings import EmbeddingService
from src.qdrant_storage import QdrantStorage
from src.search_evaluation import SearchEvaluator
from src.clustering_service import ClusteringService

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
    embedding_generator = embedding_service.embed_corpus(corpus_path, qrels_path, query_path)
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
    
    return evaluator.calculate_aggregate_metrics(), queries

def cluster_queries(queries: Dict, model_name: str, min_cluster_size: int = 5):
    """Cluster queries using HDBSCAN to find similar queries"""
    logger.info(f"Starting query clustering for {len(queries)} queries")
    
    # Initialize clustering service
    clustering_service = ClusteringService(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric='euclidean',
        cluster_selection_epsilon=0.0
    )
    
    # Load query embeddings
    clustering_service.load_query_embeddings(queries)
    
    # Perform clustering
    clustering_results = clustering_service.cluster_queries(prediction_data=True)
    
    # Analyze cluster coherence
    coherence_scores = clustering_service.analyze_cluster_coherence()
    clustering_results['cluster_coherence'] = coherence_scores
    
    # Get outlier scores
    outlier_scores = clustering_service.get_outlier_scores()
    clustering_results['outlier_scores'] = outlier_scores
    
    # Save results
    output_path = f"data/query_clusters.{model_name.replace('/', '_')}.json"
    clustering_service.save_clustering_results(output_path, include_embeddings=False)
    
    # Print summary
    print("\n" + "="*50)
    print("QUERY CLUSTERING SUMMARY")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Total Queries: {clustering_results['total_queries']}")
    print(f"Number of Clusters: {clustering_results['n_clusters']}")
    print(f"Noise Points: {clustering_results['n_noise_points']}")
    print(f"Average Cluster Size: {clustering_results['avg_cluster_size']:.2f}")
    print(f"Max Cluster Size: {clustering_results['max_cluster_size']}")
    print(f"Min Cluster Size: {clustering_results['min_cluster_size']}")
    print("-"*50)
    
    # Print top coherent clusters
    if coherence_scores:
        sorted_coherence = sorted(coherence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 Most Coherent Clusters:")
        for cluster_id, score in sorted_coherence:
            n_queries = len(clustering_results['clusters'][str(cluster_id)])
            print(f"  Cluster {cluster_id}: {score:.4f} coherence ({n_queries} queries)")
    
    print("="*50)
    
    return clustering_results

def main():
    parser = argparse.ArgumentParser(description='BEIR Corpus Embedding and Search Evaluation')
    parser.add_argument('--models', nargs='+', help='List of embedding models to use')
    parser.add_argument('--embed-only', action='store_true', help='Only embed and index, skip search')
    parser.add_argument('--search-only', action='store_true', help='Only search, skip embedding')
    parser.add_argument('--collection', type=str, help='Collection name for search-only mode')
    parser.add_argument('--cluster-queries', action='store_true', help='Perform query clustering')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='Minimum cluster size for HDBSCAN')
    
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
                metrics, queries = search_and_evaluate(
                    model_name=model_name,
                    collection_name=collection_name,
                    embedding_service=embedding_service,
                    qdrant_storage=qdrant_storage
                )
                all_metrics[model_name] = metrics
                
                # Perform query clustering if requested
                if args.cluster_queries:
                    cluster_results = cluster_queries(
                        queries=queries,
                        model_name=model_name,
                        min_cluster_size=args.min_cluster_size
                    )
            
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