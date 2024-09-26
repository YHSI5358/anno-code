## STARLET: A Framework for Aggregating Complex Medical Image Annotations

This repository contains the code for the paper "**STARLET: A Framework for Aggregating Complex Medical Image Annotations**". The framework is designed to aggregate and evaluate medical image annotations.

### Directory Structure

```
Aggregation.ipynb
datasets/
	COCO/
		GroundTruth.csv
		WorkerData.csv
	Medical/
		GroundTruth.csv
		WorkerData.csv
README.md
utils/
	aggregate.py
	data_process.py
	eval.py
```

### Files and Directories

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
  - `Technical report.pdf`: The technical report of our work.

### Usage

1. **Data Preparation**: Place your dataset files in the appropriate directories under 

datasets.

2. **Running the Notebook**: Open `Aggregation.ipynb` in Jupyter Notebook and run the cells to perform data aggregation and analysis.

### Note

​	We open-sourced the Cytology dataset and COCO dataset mentioned in the paper. However, due to business concern, we are unable to publicly share the RetinaNet training dataset in the paper.
