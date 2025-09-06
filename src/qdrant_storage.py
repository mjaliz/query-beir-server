from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, 
    Distance, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    ScoredPoint
)
from uuid import uuid4
import logging
from tqdm import tqdm
import time
from qdrant_client.http.exceptions import ResponseHandlingException
import socket

logger = logging.getLogger(__name__)

class QdrantStorage:
    def __init__(self, host: str = "localhost", port: int = 6333, 
                 api_key: Optional[str] = None, https: bool = False, 
                 max_retries: int = 3, retry_delay: int = 5):
        self.host = host
        self.port = port
        self.api_key = api_key
        self.https = https
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = None
        
        self._connect_with_retry()
    
    def _connect_with_retry(self):
        """Connect to Qdrant with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Attempting to connect to Qdrant at {self.host}:{self.port} (attempt {attempt + 1}/{self.max_retries})")
                
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    https=self.https,
                    timeout=10  # 10 second timeout for connection
                )
                
                # Test the connection by getting collections
                _ = self.client.get_collections()
                logger.info(f"Successfully connected to Qdrant at {self.host}:{self.port}")
                return
                
            except (socket.error, socket.gaierror, ResponseHandlingException, Exception) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Failed to connect to Qdrant (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect to Qdrant after {self.max_retries} attempts")
                    logger.error(f"Please ensure Qdrant is running at {self.host}:{self.port}")
                    logger.error("You can start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
                    raise ConnectionError(f"Cannot connect to Qdrant at {self.host}:{self.port}: {e}")
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """Execute a function with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (socket.error, socket.gaierror, ResponseHandlingException) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    # Try to reconnect
                    self._connect_with_retry()
                else:
                    logger.error(f"Operation failed after {self.max_retries} attempts: {e}")
                    raise

    def create_collection(self, collection_name: str, vector_size: int):
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                logger.info(f"Collection {collection_name} already exists, recreating...")
                self.client.delete_collection(collection_name)
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created collection {collection_name} with vector size {vector_size}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def get_collection_name(self, model_name: str) -> str:
        # Clean model name for collection naming
        collection_name = model_name.replace("/", "_").replace("-", "_")
        return f"beir_{collection_name}"

    def index_corpus(self, collection_name: str, embedding_generator):
        points = []
        batch_size = 100
        total_indexed = 0
        
        for doc_embedding in tqdm(embedding_generator, desc=f"Indexing to {collection_name}"):
            point = PointStruct(
                id=str(uuid4()),
                vector=doc_embedding['vector'],
                payload=doc_embedding['payload']
            )
            points.append(point)
            
            if len(points) >= batch_size:
                self._execute_with_retry(
                    self.client.upsert,
                    collection_name=collection_name,
                    points=points
                )
                total_indexed += len(points)
                points = []
        
        # Index remaining points
        if points:
            self._execute_with_retry(
                self.client.upsert,
                collection_name=collection_name,
                points=points
            )
            total_indexed += len(points)
        
        logger.info(f"Indexed {total_indexed} documents to collection {collection_name}")

    def search(self, collection_name: str, query_vector: List[float], 
               top_k: int = 10, score_threshold: Optional[float] = None) -> List[ScoredPoint]:
        search_params = {
            "collection_name": collection_name,
            "query_vector": query_vector,
            "limit": top_k,
            "with_payload": True
        }
        
        if score_threshold is not None:
            search_params["score_threshold"] = score_threshold
        
        results = self._execute_with_retry(self.client.search, **search_params)
        return results

    def search_with_filter(self, collection_name: str, query_vector: List[float],
                          filter_dict: Dict[str, Any], top_k: int = 10) -> List[ScoredPoint]:
        filter_conditions = []
        for key, value in filter_dict.items():
            filter_conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
            )
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=filter_conditions),
            limit=top_k
        )
        return results

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": info.name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "config": info.config
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return None

    def delete_collection(self, collection_name: str):
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise