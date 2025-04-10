"""
Bounding box clustering module.
Implements clustering algorithms for grouping similar bounding boxes.
"""

from typing import List, Dict, Tuple
import numpy as np
from .bounding_box import BoundingBox

class BoxCluster:
    """Represents a cluster of similar bounding boxes."""
    
    def __init__(self, initial_box: BoundingBox, box_id: int):
        """
        Initialize with the first bounding box in the cluster.
        
        Args:
            initial_box: First bounding box in the cluster
            box_id: Unique identifier for the cluster
        """
        self.boxes = [initial_box]
        self.cluster_id = box_id
        self.representative = initial_box
        
    def add_box(self, box: BoundingBox) -> None:
        """Add a new bounding box to the cluster."""
        self.boxes.append(box)
        
    def calculate_representative(self, method: str = 'average') -> BoundingBox:
        """
        Calculate representative box for the cluster.
        
        Args:
            method: Method to calculate representative ('average' or 'median')
            
        Returns:
            Representative bounding box
        """
        if not self.boxes:
            return None
            
        coords = np.array([box.coords for box in self.boxes])
        
        if method == 'average':
            avg_coords = np.mean(coords, axis=0)
        elif method == 'median':
            avg_coords = np.median(coords, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        self.representative = BoundingBox(avg_coords.tolist())
        return self.representative
        
    def __len__(self) -> int:
        return len(self.boxes)

class BoxClusterer:
    """Handles clustering of bounding boxes based on similarity."""
    
    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize with IOU threshold for clustering.
        
        Args:
            iou_threshold: Minimum IOU for boxes to be in same cluster
        """
        self.iou_threshold = iou_threshold
        self.clusters: List[BoxCluster] = []
        
    def cluster_boxes(self, boxes: List[BoundingBox]) -> List[BoxCluster]:
        """
        Cluster bounding boxes based on IOU similarity.
        
        Args:
            boxes: List of bounding boxes to cluster
            
        Returns:
            List of BoxCluster objects
        """
        self.clusters = []
        
        for i, box in enumerate(boxes):
            matched = False
            for cluster in self.clusters:
                if box.calculate_iou(cluster.representative) >= self.iou_threshold:
                    cluster.add_box(box)
                    matched = True
                    break
                    
            if not matched:
                new_cluster = BoxCluster(box, len(self.clusters))
                self.clusters.append(new_cluster)
                
        # Update representatives for all clusters
        for cluster in self.clusters:
            cluster.calculate_representative()
            
        return self.clusters
        
    def get_cluster_stats(self) -> Dict:
        """
        Get statistics about the clusters.
        
        Returns:
            Dictionary with cluster statistics
        """
        return {
            'num_clusters': len(self.clusters),
            'avg_cluster_size': np.mean([len(c) for c in self.clusters]) if self.clusters else 0,
            'max_cluster_size': max([len(c) for c in self.clusters]) if self.clusters else 0
        }
