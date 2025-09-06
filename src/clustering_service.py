import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from collections import defaultdict
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class ClusteringService:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = None,
                 metric: str = 'euclidean', cluster_selection_epsilon: float = 0.0,
                 cluster_selection_method: str = 'eom', alpha: float = 1.0,
                 algorithm: str = 'best'):
        """
        Initialize HDBSCAN clustering service for query similarity analysis.
        
        Args:
            min_cluster_size: Minimum size of clusters (increase for larger clusters)
            min_samples: Minimum samples in neighborhood (None = min_cluster_size)
            metric: Distance metric ('euclidean', 'cosine', 'manhattan')
            cluster_selection_epsilon: Cut distance for extracting flat clusters
            cluster_selection_method: 'eom' (default) or 'leaf' (smaller, tighter clusters)
            alpha: Conservative parameter for 'eom' (>1.0 = more conservative/smaller)
            algorithm: Algorithm to use ('best', 'prims_kdtree', 'prims_balltree', 'boruvka_kdtree')
        
        For smaller, more accurate clusters:
        - Use smaller min_cluster_size (3-10)
        - Use 'leaf' cluster_selection_method
        - Increase alpha (1.5-2.0) for 'eom' method
        - Use smaller min_samples (1-3)
        - Use 'cosine' metric for text embeddings
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_method = cluster_selection_method
        self.alpha = alpha
        self.algorithm = algorithm
        self.clusterer = None
        self.query_embeddings = {}
        self.query_texts = {}
        self.cluster_labels = None
        
    def load_query_embeddings(self, queries: Dict[str, Dict[str, Any]]):
        """
        Load query embeddings and texts from the queries dictionary.
        
        Args:
            queries: Dictionary with query_id as key and dict containing 'vector' and 'text'
        """
        self.query_embeddings = {}
        self.query_texts = {}
        
        for query_id, query_data in queries.items():
            self.query_embeddings[query_id] = np.array(query_data['vector'])
            self.query_texts[query_id] = query_data['text']
        
        logger.info(f"Loaded {len(self.query_embeddings)} query embeddings")
    
    def cluster_queries(self, prediction_data: bool = True) -> Dict[str, Any]:
        """
        Perform HDBSCAN clustering on query embeddings.
        
        Args:
            prediction_data: Whether to generate prediction data for soft clustering
            
        Returns:
            Dictionary containing clustering results and statistics
        """
        if not self.query_embeddings:
            raise ValueError("No query embeddings loaded. Call load_query_embeddings first.")
        
        # Convert embeddings to matrix
        query_ids = list(self.query_embeddings.keys())
        embedding_matrix = np.array([self.query_embeddings[qid] for qid in query_ids])
        
        logger.info(f"Starting HDBSCAN clustering on {len(query_ids)} queries...")
        
        # Initialize and fit HDBSCAN
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method=self.cluster_selection_method,
            alpha=self.alpha,
            algorithm=self.algorithm,
            prediction_data=prediction_data
        )
        
        self.cluster_labels = self.clusterer.fit_predict(embedding_matrix)
        
        # Organize results
        clusters = defaultdict(list)
        noise_points = []
        
        for idx, (query_id, label) in enumerate(zip(query_ids, self.cluster_labels)):
            if label == -1:
                noise_points.append(query_id)
            else:
                clusters[int(label)].append(query_id)
        
        # Calculate statistics
        n_clusters = len(clusters)
        n_noise = len(noise_points)
        cluster_sizes = [len(cluster) for cluster in clusters.values()]
        
        results = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'total_queries': len(query_ids),
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'clusters': dict(clusters),
            'noise_points': noise_points
        }
        
        logger.info(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points")
        
        return results
    
    def find_similar_queries_in_cluster(self, cluster_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find the most similar queries within a specific cluster.
        
        Args:
            cluster_id: The cluster ID to analyze
            top_k: Number of most similar pairs to return
            
        Returns:
            List of similar query pairs with similarity scores
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed yet. Call cluster_queries first.")
        
        # Get query IDs in this cluster
        query_ids = list(self.query_embeddings.keys())
        cluster_query_ids = [
            qid for qid, label in zip(query_ids, self.cluster_labels) 
            if label == cluster_id
        ]
        
        if len(cluster_query_ids) < 2:
            return []
        
        # Calculate pairwise similarities
        embeddings = np.array([self.query_embeddings[qid] for qid in cluster_query_ids])
        similarity_matrix = cosine_similarity(embeddings)
        
        # Extract top-k similar pairs
        similar_pairs = []
        n = len(cluster_query_ids)
        
        for i in range(n):
            for j in range(i + 1, n):
                similar_pairs.append({
                    'query_id_1': cluster_query_ids[i],
                    'query_text_1': self.query_texts[cluster_query_ids[i]],
                    'query_id_2': cluster_query_ids[j],
                    'query_text_2': self.query_texts[cluster_query_ids[j]],
                    'similarity': float(similarity_matrix[i, j])
                })
        
        # Sort by similarity and return top-k
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_pairs[:top_k]
    
    def get_cluster_representatives(self, n_representatives: int = 3) -> Dict[int, List[str]]:
        """
        Get representative queries for each cluster (closest to cluster center).
        
        Args:
            n_representatives: Number of representative queries per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of representative query_ids
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed yet. Call cluster_queries first.")
        
        representatives = {}
        query_ids = list(self.query_embeddings.keys())
        
        # For each cluster
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
            
            # Get queries in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_query_ids = [qid for qid, mask in zip(query_ids, cluster_mask) if mask]
            cluster_embeddings = np.array([
                self.query_embeddings[qid] for qid in cluster_query_ids
            ])
            
            # Calculate cluster center
            cluster_center = np.mean(cluster_embeddings, axis=0)
            
            # Find queries closest to center
            distances = np.linalg.norm(cluster_embeddings - cluster_center, axis=1)
            closest_indices = np.argsort(distances)[:n_representatives]
            
            representatives[int(cluster_id)] = [
                cluster_query_ids[idx] for idx in closest_indices
            ]
        
        return representatives
    
    def save_clustering_results(self, output_path: str, include_embeddings: bool = False):
        """
        Save clustering results to a JSON file.
        
        Args:
            output_path: Path to save the results
            include_embeddings: Whether to include embeddings in the output
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed yet. Call cluster_queries first.")
        
        # Get clustering results
        results = self.cluster_queries(prediction_data=False)
        
        # Add representative queries
        representatives = self.get_cluster_representatives()
        results['cluster_representatives'] = {}
        
        for cluster_id, rep_ids in representatives.items():
            results['cluster_representatives'][str(cluster_id)] = [
                {
                    'query_id': qid,
                    'query_text': self.query_texts[qid]
                }
                for qid in rep_ids
            ]
        
        # Add similar queries for each cluster
        results['similar_queries_per_cluster'] = {}
        for cluster_id in results['clusters'].keys():
            similar_pairs = self.find_similar_queries_in_cluster(int(cluster_id), top_k=5)
            results['similar_queries_per_cluster'][str(cluster_id)] = similar_pairs
        
        # Add query-to-cluster mapping
        query_ids = list(self.query_embeddings.keys())
        results['query_cluster_mapping'] = {
            qid: int(label) if label != -1 else 'noise'
            for qid, label in zip(query_ids, self.cluster_labels)
        }
        
        # Optionally include embeddings
        if include_embeddings:
            results['query_embeddings'] = {
                qid: embedding.tolist() 
                for qid, embedding in self.query_embeddings.items()
            }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved clustering results to {output_path}")
    
    def analyze_cluster_coherence(self) -> Dict[int, float]:
        """
        Analyze the coherence of each cluster using average pairwise similarity.
        
        Returns:
            Dictionary mapping cluster_id to average coherence score
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering not performed yet. Call cluster_queries first.")
        
        coherence_scores = {}
        query_ids = list(self.query_embeddings.keys())
        
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:  # Skip noise points
                continue
            
            # Get queries in this cluster
            cluster_query_ids = [
                qid for qid, label in zip(query_ids, self.cluster_labels) 
                if label == cluster_id
            ]
            
            if len(cluster_query_ids) < 2:
                coherence_scores[int(cluster_id)] = 1.0
                continue
            
            # Calculate average pairwise similarity
            embeddings = np.array([self.query_embeddings[qid] for qid in cluster_query_ids])
            similarity_matrix = cosine_similarity(embeddings)
            
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(similarity_matrix, k=1)
            n_pairs = np.sum(upper_triangle > 0)
            
            if n_pairs > 0:
                avg_similarity = np.sum(upper_triangle) / n_pairs
            else:
                avg_similarity = 1.0
            
            coherence_scores[int(cluster_id)] = float(avg_similarity)
        
        return coherence_scores
    
    def get_outlier_scores(self) -> Dict[str, float]:
        """
        Get outlier scores for all queries from HDBSCAN.
        
        Returns:
            Dictionary mapping query_id to outlier score
        """
        if self.clusterer is None:
            raise ValueError("Clustering not performed yet. Call cluster_queries first.")
        
        query_ids = list(self.query_embeddings.keys())
        outlier_scores = self.clusterer.outlier_scores_
        
        return {
            qid: float(score) 
            for qid, score in zip(query_ids, outlier_scores)
        }