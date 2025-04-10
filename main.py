"""
Main entry point for bounding box aggregation system.
Provides command line interface for processing and evaluating bounding boxes.
"""

import argparse
import json
import pandas as pd
from typing import Dict, List, Tuple
from core.bounding_box import BoundingBox
from core.processor import BoxProcessor

def load_data(worker_data_path: str, ground_truth_path: str) -> Tuple:
    """
    Load worker data and ground truth from CSV files.
    
    Args:
        worker_data_path: Path to worker data CSV
        ground_truth_path: Path to ground truth CSV
        
    Returns:
        Tuple containing (worker_boxes, worker_categories, worker_ids, 
                          ground_truth_boxes, ground_truth_categories)
    """
    # Load worker data
    worker_df = pd.read_csv(worker_data_path)
    worker_boxes = [
        BoundingBox(row['x1'], row['y1'], row['x2'], row['y2'])
        for _, row in worker_df.iterrows()
    ]
    worker_categories = worker_df['category'].tolist()
    worker_ids = worker_df['worker_id'].tolist()
    
    # Load ground truth
    gt_df = pd.read_csv(ground_truth_path)
    ground_truth_boxes = [
        BoundingBox(row['x1'], row['y1'], row['x2'], row['y2'])
        for _, row in gt_df.iterrows()
    ]
    ground_truth_categories = gt_df['category'].tolist()
    
    return (
        worker_boxes, 
        worker_categories, 
        worker_ids,
        ground_truth_boxes, 
        ground_truth_categories
    )

def process_and_evaluate(
    worker_data_path: str,
    ground_truth_path: str,
    iou_threshold: float = 0.5,
    aggregation_strategy: str = 'average',
    weights: List[float] = None
) -> Dict:
    """
    Process worker data and evaluate against ground truth.
    
    Args:
        worker_data_path: Path to worker data CSV
        ground_truth_path: Path to ground truth CSV
        iou_threshold: IOU threshold for clustering
        aggregation_strategy: Aggregation strategy ('average' or 'weighted')
        weights: Optional weights for weighted aggregation
        
    Returns:
        Dictionary containing processing and evaluation results
    """
    # Load data
    worker_boxes, worker_cats, worker_ids, gt_boxes, gt_cats = load_data(
        worker_data_path, ground_truth_path
    )
    
    # Process boxes
    processor = BoxProcessor(
        iou_threshold=iou_threshold,
        aggregation_strategy=aggregation_strategy,
        weights=weights
    )
    
    # Get aggregated results
    results = processor.process_boxes(worker_boxes, worker_cats)
    
    # Evaluate against ground truth
    evaluation = processor.evaluate_results(
        results['aggregated_boxes'],
        gt_boxes,
        results['categories'],
        gt_cats
    )
    
    # Calculate worker quality
    worker_quality = processor.get_worker_quality_scores(
        worker_ids,
        gt_boxes,
        gt_cats
    )
    
    return {
        'processing': results,
        'evaluation': evaluation,
        'worker_quality': worker_quality
    }

def main():
    """Command line interface for bounding box aggregation."""
    parser = argparse.ArgumentParser(
        description='Aggregate and evaluate bounding box annotations'
    )
    parser.add_argument(
        'worker_data',
        help='Path to CSV file containing worker bounding box data'
    )
    parser.add_argument(
        'ground_truth',
        help='Path to CSV file containing ground truth bounding boxes'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5,
        help='IOU threshold for clustering (default: 0.5)'
    )
    parser.add_argument(
        '--strategy',
        choices=['average', 'weighted'],
        default='average',
        help='Aggregation strategy (default: average)'
    )
    parser.add_argument(
        '--output',
        default='results.json',
        help='Output file path (default: results.json)'
    )
    
    args = parser.parse_args()
    
    # Process and evaluate
    results = process_and_evaluate(
        args.worker_data,
        args.ground_truth,
        iou_threshold=args.iou_threshold,
        aggregation_strategy=args.strategy
    )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
