"""
Bounding box aggregation module.
Implements various strategies for aggregating clustered bounding boxes.
"""

from typing import List, Dict, Optional
import numpy as np
from .bounding_box import BoundingBox
from .clustering import BoxCluster

class AggregationStrategy:
    """Abstract base class for aggregation strategies."""
    
    def aggregate(self, cluster: BoxCluster) -> BoundingBox:
        """
        Aggregate boxes in a cluster to produce final box.
        
        Args:
            cluster: BoxCluster object containing boxes to aggregate
            
        Returns:
            Aggregated bounding box
        """
        raise NotImplementedError

class AverageAggregation(AggregationStrategy):
    """Aggregates boxes by averaging coordinates."""
    
    def aggregate(self, cluster: BoxCluster) -> BoundingBox:
        coords = np.array([box.coords for box in cluster.boxes])
        avg_coords = np.mean(coords, axis=0)
        return BoundingBox(avg_coords.tolist())

class WeightedAggregation(AggregationStrategy):
    """Aggregates boxes with worker confidence weights."""
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize with optional weights.
        
        Args:
            weights: List of weights for each box (default: equal weights)
        """
        self.weights = weights
        
    def aggregate(self, cluster: BoxCluster) -> BoundingBox:
        coords = np.array([box.coords for box in cluster.boxes])
        
        if self.weights is None or len(self.weights) != len(cluster.boxes):
            weights = np.ones(len(cluster.boxes))
        else:
            weights = np.array(self.weights)
            
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        weighted_coords = np.sum(coords * weights[:, np.newaxis], axis=0)
        return BoundingBox(weighted_coords.tolist())

class BoxAggregator:
    """Handles aggregation of clustered bounding boxes."""
    
    def __init__(self, strategy: str = 'average', weights: Optional[List[float]] = None):
        """
        Initialize with aggregation strategy.
        
        Args:
            strategy: Aggregation strategy ('average' or 'weighted')
            weights: Optional weights for weighted aggregation
        """
        self.strategy = self._get_strategy(strategy, weights)
        
    def _get_strategy(self, strategy: str, weights: Optional[List[float]]) -> AggregationStrategy:
        """Get appropriate aggregation strategy instance."""
        if strategy == 'average':
            return AverageAggregation()
        elif strategy == 'weighted':
            return WeightedAggregation(weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
            
    def aggregate_clusters(self, clusters: List[BoxCluster]) -> List[BoundingBox]:
        """
        Aggregate all clusters to produce final boxes.
        
        Args:
            clusters: List of BoxCluster objects
            
        Returns:
            List of aggregated bounding boxes
        """
        return [self.strategy.aggregate(cluster) for cluster in clusters]
        
    def calculate_worker_agreement(self, cluster: BoxCluster) -> float:
        """
        Calculate agreement level among workers for a cluster.
        
        Args:
            cluster: BoxCluster object
            
        Returns:
            Agreement score between 0 and 1
        """
        if len(cluster.boxes) < 2:
            return 1.0
            
        ious = []
        for i in range(len(cluster.boxes)):
            for j in range(i+1, len(cluster.boxes)):
                iou = cluster.boxes[i].calculate_iou(cluster.boxes[j])
                ious.append(iou)
                
        return np.mean(ious) if ious else 0.0
