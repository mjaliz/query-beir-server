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
                        qdrant_storage: QdrantStorage,
                        queries: dict = None):
    """Search queries and evaluate results"""
    logger.info(f"Starting search and evaluation for collection: {collection_name}")
    
    # Initialize evaluator
    evaluator = SearchEvaluator(qdrant_storage, collection_name)
    
    # Load relevance judgments
    evaluator.load_qrels(config.data.qrels_path)
    
    # Embed queries if not provided
    if queries is None:
        queries = embedding_service.embed_queries(config.data.query_path)
    
    # Search and evaluate
    results = evaluator.search_queries(queries, top_k=config.search.top_k)
    
    # Save false positives
    false_positives_path = f"{config.data.false_positives_output}.{model_name.replace('/', '_')}.jsonl"
    evaluator.save_false_positives(false_positives_path)
    
    # Print summary
    evaluator.print_summary()
    
    return evaluator.calculate_aggregate_metrics()

def get_query_embeddings(model_name: str) -> dict:
    """Get query embeddings for clustering or other purposes"""
    logger.info(f"Loading query embeddings with model: {model_name}")
    
    # Initialize embedding service
    embedding_service = EmbeddingService(
        model_name=model_name,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size
    )
    
    # Embed queries
    queries = embedding_service.embed_queries(config.data.query_path)
    logger.info(f"Embedded {len(queries)} queries")
    
    return queries

def cluster_queries(queries: dict, model_name: str, **kwargs):
    """Cluster queries using HDBSCAN to find similar queries"""
    logger.info(f"Starting query clustering for {len(queries)} queries")
    
    # Extract clustering parameters with defaults
    min_cluster_size = kwargs.get('min_cluster_size', 5)
    min_samples = kwargs.get('min_samples', None)
    metric = kwargs.get('metric', 'cosine')
    cluster_method = kwargs.get('cluster_method', 'eom')
    alpha = kwargs.get('alpha', 1.0)
    epsilon = kwargs.get('epsilon', 0.0)
    
    logger.info(f"Clustering params: min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
                f"metric={metric}, method={cluster_method}, alpha={alpha}, epsilon={epsilon}")
    
    # Initialize clustering service
    clustering_service = ClusteringService(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=epsilon,
        cluster_selection_method=cluster_method,
        alpha=alpha
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
            # Try both integer and string keys for compatibility
            if cluster_id in clustering_results['clusters']:
                n_queries = len(clustering_results['clusters'][cluster_id])
            elif str(cluster_id) in clustering_results['clusters']:
                n_queries = len(clustering_results['clusters'][str(cluster_id)])
            else:
                n_queries = 0
            print(f"  Cluster {cluster_id}: {score:.4f} coherence ({n_queries} queries)")
    
    print("="*50)
    
    return clustering_results

def main():
    parser = argparse.ArgumentParser(description='BEIR Corpus Embedding and Search Evaluation')
    parser.add_argument('--models', nargs='+', help='List of embedding models to use')
    parser.add_argument('--embed-only', action='store_true', help='Only embed and index, skip search')
    parser.add_argument('--search-only', action='store_true', help='Only search, skip embedding')
    parser.add_argument('--cluster-only', action='store_true', help='Only perform query clustering')
    parser.add_argument('--collection', type=str, help='Collection name for search-only mode')
    parser.add_argument('--cluster-queries', action='store_true', help='Perform query clustering after search')
    parser.add_argument('--min-cluster-size', type=int, default=5, help='Minimum cluster size (3-10 for smaller clusters)')
    parser.add_argument('--min-samples', type=int, default=None, help='Min samples in neighborhood (1-3 for tighter clusters)')
    parser.add_argument('--cluster-method', type=str, default='eom', choices=['eom', 'leaf'], 
                       help='Cluster selection method: eom (default) or leaf (smaller clusters)')
    parser.add_argument('--cluster-metric', type=str, default='cosine', choices=['euclidean', 'cosine', 'manhattan'],
                       help='Distance metric for clustering (cosine recommended for text)')
    parser.add_argument('--cluster-alpha', type=float, default=1.0, 
                       help='Alpha parameter for eom method (1.5-2.0 for smaller clusters)')
    parser.add_argument('--cluster-epsilon', type=float, default=0.0,
                       help='Epsilon for flat clustering (0.1-0.5 for tighter clusters)')
    
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
            
            # Handle cluster-only mode
            if args.cluster_only:
                queries = get_query_embeddings(model_name)
                cluster_results = cluster_queries(
                    queries=queries,
                    model_name=model_name,
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    metric=args.cluster_metric,
                    cluster_method=args.cluster_method,
                    alpha=args.cluster_alpha,
                    epsilon=args.cluster_epsilon
                )
                continue
            
            # Handle embed and search modes
            queries = None  # Will be loaded when needed
            
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
                # Get query embeddings first if clustering is requested
                if args.cluster_queries:
                    queries = get_query_embeddings(model_name)
                
                # Search and evaluate
                metrics = search_and_evaluate(
                    model_name=model_name,
                    collection_name=collection_name,
                    embedding_service=embedding_service,
                    qdrant_storage=qdrant_storage,
                    queries=queries
                )
                all_metrics[model_name] = metrics
                
                # Perform query clustering if requested
                if args.cluster_queries:
                    cluster_results = cluster_queries(
                        queries=queries,
                        model_name=model_name,
                        min_cluster_size=args.min_cluster_size,
                        min_samples=args.min_samples,
                        metric=args.cluster_metric,
                        cluster_method=args.cluster_method,
                        alpha=args.cluster_alpha,
                        epsilon=args.cluster_epsilon
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