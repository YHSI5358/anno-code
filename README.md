## STARLET: A Framework for Aggregating Complex Medical Image Annotations

This repository contains the code for the paper "**STARLET: A Framework for Aggregating Complex Medical Image Annotations**". The framework is designed to aggregate and evaluate medical image annotations.

## Simple Implementation

- **Aggregation.ipynb**: Jupyter notebook containing the main code for data aggregation and analysis.
- **datasets/**: Directory containing datasets used in the framework.
  - **COCO/**: Contains COCO dataset files.
    - `GroundTruth.csv`: Ground truth annotations for the COCO dataset.
    - `WorkerData.csv`: Worker annotations for the COCO dataset.
  - **Medical/**: Contains medical dataset files.
    - `GroundTruth.csv`: Ground truth annotations for the medical dataset.
    - `WorkerData.csv`: Worker annotations for the medical dataset.
- **utils/**: Directory containing utility scripts.
  - `aggregate.py`: Script for aggregating annotations.
  - `data_process.py`: Script for processing data.
  - `eval.py`: Script for evaluating the aggregated annotations.
- **TechnicalReport/**: Our technical report.
  - `TechnicalReport.pdf`: The technical report of our work.

### Usage

1. **Data Preparation**: Place your dataset files in the appropriate directories under 

datasets.

2. **Running the Notebook**: Open `Aggregation.ipynb` in Jupyter Notebook and run the cells to perform data aggregation and analysis.

### Note

​	We open-sourced the Cytology dataset and COCO dataset mentioned in the paper. However, due to business concern, we are unable to publicly share the RetinaNet training dataset in the paper.


### Characteristics:
- Single-file implementations
- Minimal error handling
- Basic functionality only



## Core Architecture 

Located in `core/` directory with modular components:

### Core Modules:
1. `bounding_box.py`: 
   - Robust BoundingBox class with IOU calculation
   - Type hints and validation

2. `clustering.py`:
   - Advanced clustering algorithms
   - Configurable IOU thresholds

3. `aggregation.py`:
   - Multiple strategies (average, weighted)
   - Worker quality weighting

4. `evaluation.py`:
   - Comprehensive metrics (precision, recall, F1)
   - Detailed performance analysis

5. `processor.py`:
   - Main processing pipeline
   - End-to-end workflow


### Example Usage:
```python
from core.processor import BoxProcessor

# Initialize with custom parameters
processor = BoxProcessor(
    iou_threshold=0.6,
    aggregation_strategy='weighted'
)

# Process boxes and evaluate
results = processor.process_boxes(worker_boxes, worker_categories)
evaluation = processor.evaluate_results(results, ground_truth)

# Get worker quality scores
quality_scores = processor.get_worker_quality_scores()
```

## Command Line Interface

The new `main.py` provides a convenient CLI:

```bash
python main.py WorkerData.csv GroundTruth.csv \
    --iou_threshold 0.6 \
    --strategy weighted \
    --output results.json
```
