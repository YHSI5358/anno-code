"""
Main processing module.
Coordinates the complete bounding box aggregation pipeline.
"""

from typing import List, Dict, Optional
import numpy as np
from .bounding_box import BoundingBox
from .clustering import BoxClusterer
from .aggregation import BoxAggregator
from .evaluation import EvaluationMetrics

class BoxProcessor:
    """Main class for processing bounding box aggregation."""
    
    def __init__(
        self,
        iou_threshold: float = 0.5,
        aggregation_strategy: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize the processor with configuration.
        
        Args:
            iou_threshold: IOU threshold for clustering
            aggregation_strategy: Strategy for aggregation ('average' or 'weighted')
            weights: Optional weights for weighted aggregation
        """
        self.clusterer = BoxClusterer(iou_threshold)
        self.aggregator = BoxAggregator(aggregation_strategy, weights)
        self.evaluator = EvaluationMetrics(iou_threshold)
        
    def process_boxes(
        self,
        boxes: List[BoundingBox],
        categories: Optional[List[str]] = None
    ) -> Dict:
        """
        Process a list of bounding boxes through the full pipeline.
        
        Args:
            boxes: List of bounding boxes to process
            categories: Optional list of categories for each box
            
        Returns:
            Dictionary containing processing results
        """
        if categories is None:
            categories = ['object'] * len(boxes)
            
        if len(boxes) != len(categories):
            raise ValueError("Boxes and categories must have same length")
            
        # Cluster the boxes
        clusters = self.clusterer.cluster_boxes(boxes)
        
        # Aggregate the clusters
        aggregated_boxes = self.aggregator.aggregate_clusters(clusters)
        
        # Calculate cluster statistics
        cluster_stats = self.clusterer.get_cluster_stats()
        
        # Calculate worker agreement
        agreement_scores = [
            self.aggregator.calculate_worker_agreement(cluster)
            for cluster in clusters
        ]
        
        return {
            'clusters': clusters,
            'aggregated_boxes': aggregated_boxes,
            'cluster_stats': cluster_stats,
            'agreement_scores': agreement_scores,
            'categories': categories
        }
        
    def evaluate_results(
        self,
        predicted_boxes: List[BoundingBox],
        ground_truth_boxes: List[BoundingBox],
        predicted_categories: List[str],
        ground_truth_categories: List[str]
    ) -> Dict:
        """
        Evaluate predicted boxes against ground truth.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth boxes
            predicted_categories: List of predicted categories
            ground_truth_categories: List of ground truth categories
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate standard metrics
        standard_metrics = self.evaluator.calculate_one_to_one_metrics(
            predicted_boxes,
            ground_truth_boxes,
            predicted_categories,
            ground_truth_categories
        )
        
        # Calculate Braylan metrics
        braylan_metrics = self.evaluator.calculate_braylan_metrics(
            predicted_boxes,
            ground_truth_boxes,
            predicted_categories,
            ground_truth_categories
        )
        
        return {
            'standard_metrics': standard_metrics,
            'braylan_metrics': braylan_metrics
        }
        
    def get_worker_quality_scores(
        self,
        worker_ids: List[str],
        ground_truth_boxes: List[BoundingBox],
        ground_truth_categories: List[str]
    ) -> Dict[str, float]:
        """
        Calculate quality scores for each worker.
        
        Args:
            worker_ids: List of worker IDs corresponding to boxes
            ground_truth_boxes: List of ground truth boxes
            ground_truth_categories: List of ground truth categories
            
        Returns:
            Dictionary mapping worker IDs to quality scores
        """
        unique_workers = list(set(worker_ids))
        worker_scores = {worker: [] for worker in unique_workers}
        
        for worker, box, cat in zip(worker_ids, ground_truth_boxes, ground_truth_categories):
            max_iou = 0
            for gt_box, gt_cat in zip(ground_truth_boxes, ground_truth_categories):
                if cat == gt_cat:
                    iou = box.calculate_iou(gt_box)
                    if iou > max_iou:
                        max_iou = iou
            worker_scores[worker].append(max_iou)
            
        return {
            worker: np.mean(scores) if scores else 0
            for worker, scores in worker_scores.items()
        }
