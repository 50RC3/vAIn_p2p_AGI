import logging
import numpy as np
import torch
from typing import List, Optional, Dict
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class UnsupervisedLearningModule:
    """Unsupervised learning module for clustering and pattern discovery"""
    
    def __init__(self, embedding_dim: int = 768, n_clusters: int = 10, buffer_size: int = 1000):
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.buffer = []
        self.buffer_size = buffer_size
        self.kmeans = None
        self.pca = None
        self.is_initialized = False
        self.cluster_examples = {i: [] for i in range(n_clusters)}
        self.cluster_counts = {i: 0 for i in range(n_clusters)}
        self.cluster_centroids = None
        
    def initialize(self) -> bool:
        """Initialize clustering models"""
        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
            
            self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.pca = PCA(n_components=min(50, self.embedding_dim))
            self.is_initialized = True
            logger.info(f"Initialized unsupervised module with {self.n_clusters} clusters")
            return True
        except ImportError:
            logger.error("sklearn not available for unsupervised learning")
            return False
            
    def add_to_buffer(self, example):
        """Add embedding or text to processing buffer"""
        # Handle various input types
        if isinstance(example, str):
            # Just store the string for later processing
            self.buffer.append(example)
        elif isinstance(example, torch.Tensor):
            # Convert tensor to numpy array
            example_array = example.detach().cpu().numpy()
            if example_array.ndim == 1:
                self.buffer.append(example_array)
            else:
                # Handle batches by adding each item
                for i in range(example_array.shape[0]):
                    self.buffer.append(example_array[i])
        elif isinstance(example, np.ndarray):
            # Add numpy array directly
            self.buffer.append(example)
        
        # Trim buffer if it's too large
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    async def update_clusters(self) -> bool:
        """Update clusters based on current buffer data"""
        if not self.is_initialized or len(self.buffer) < self.n_clusters:
            return False
            
        try:
            # Process buffer items to get embeddings
            processed_buffer = []
            for item in self.buffer:
                if isinstance(item, str):
                    # Skip for now, will be processed when embeddings are available
                    continue
                else:
                    # Already an embedding
                    processed_buffer.append(item)
                    
            if len(processed_buffer) < self.n_clusters:
                return False
                
            # Stack embeddings and reduce dimensionality
            embeddings = np.vstack(processed_buffer)
            reduced_data = self.pca.fit_transform(embeddings)
            
            # Update clustering model
            self.kmeans.fit(reduced_data)
            self.cluster_centroids = self.kmeans.cluster_centers_
            logger.info(f"Updated clustering with {len(processed_buffer)} examples")
            return True
        except Exception as e:
            logger.error(f"Error updating clusters: {e}")
            return False
    
    def predict_cluster(self, embedding) -> Optional[int]:
        """Predict cluster for a given embedding"""
        if not self.is_initialized or self.kmeans is None:
            logger.warning("Clustering model not initialized")
            return None
            
        try:
            # Handle tensor input
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
                
            # Reshape if needed (handle both 1D and 2D)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
                
            # Reduce dimensionality and predict
            reduced = self.pca.transform(embedding)
            cluster = self.kmeans.predict(reduced)[0]
            
            # Update cluster statistics
            self.cluster_counts[cluster] += 1
            
            return int(cluster)
        except Exception as e:
            logger.error(f"Error predicting cluster: {e}")
            return None
    
    async def add_example(self, cluster: int, example: str) -> None:
        """Add an example text to a specific cluster"""
        if cluster in self.cluster_examples:
            self.cluster_examples[cluster].append(example)
            # Keep only recent examples
            if len(self.cluster_examples[cluster]) > 50:
                self.cluster_examples[cluster] = self.cluster_examples[cluster][-50:]
                
    async def try_train_from_buffer(self) -> bool:
        """Train the clustering model from the buffer"""
        return await self.update_clusters()
        
    def get_cluster_examples(self, cluster: int) -> List[str]:
        """Get examples for a specific cluster"""
        if cluster in self.cluster_examples:
            return self.cluster_examples[cluster]
        return []
