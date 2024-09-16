# Anomaly Detection using DBSCAN

## Overview

This repository implements anomaly detection using the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm. DBSCAN is particularly effective for anomaly detection as it can identify clusters of varying shapes and sizes, while also isolating points that don't belong to any cluster as anomalies.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Examples](#examples)

## Installation

To use this project, you need Python 3.7+ and the following libraries:

```bash
pip install numpy pandas scikit-learn matplotlib
```


## Usage

1. Prepare your data in CSV format with features in columns.
2. Run the script:

```bash
python dbscan_anomaly_detection.py --input your_data.csv --eps 0.5 --min_samples 5
```

Parameters:
- `--input`: Path to your input CSV file
- `--eps`: The maximum distance between two samples for one to be considered as in the neighborhood of the other
- `--min_samples`: The number of samples in a neighborhood for a point to be considered as a core point

## How It Works

DBSCAN works by grouping together points that are closely packed together, marking as outliers points that lie alone in low-density regions. Here's a brief overview of the process:

1. The algorithm starts with an arbitrary point and finds all points within distance `eps` of it.
2. If there are at least `min_samples` points within `eps` distance, a cluster is started. Otherwise, the point is labeled as noise.
3. This process continues for all points in the dataset, growing clusters as it finds density-connected points.
4. Points that are not part of any cluster are considered anomalies.

Our implementation uses DBSCAN from scikit-learn and adds additional processing to identify and visualize anomalies.

## Examples

Here's a simple example of how to use the anomaly detection:

```python
from dbscan_anomaly_detection import DBSCANAnomalyDetector

# Create detector
detector = DBSCANAnomalyDetector(eps=0.5, min_samples=5)

# Fit and predict
anomalies = detector.fit_predict(X)

# Visualize results
detector.visualize(X, anomalies)
```

