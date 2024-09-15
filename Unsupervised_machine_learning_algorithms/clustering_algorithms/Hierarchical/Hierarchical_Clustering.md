# Hierarchical Clustering Algorithm

## Overview

This project implements hierarchical clustering, a method of cluster analysis which seeks to build a hierarchy of clusters. Hierarchical clustering is a powerful technique used in data mining and statistics for grouping similar objects into clusters.

## Features

- Agglomerative (bottom-up) clustering approach
- Multiple linkage criteria: single, complete, average, and Ward's method
- Dendrogram visualization for easy interpretation of clustering results
- Supports various distance metrics (Euclidean, Manhattan, etc.)
- Ability to handle high-dimensional data

## Installation

To use this project, clone the repository and install the required dependencies:


## Usage

Here's a basic example of how to use the hierarchical clustering implementation:

```python
from hierarchical_clustering import HierarchicalClustering
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)

# Initialize the clustering model
hc = HierarchicalClustering(n_clusters=3, linkage='ward')

# Fit the model and predict clusters
clusters = hc.fit_predict(X)

# Visualize the results
hc.plot_dendrogram()
```

For more detailed usage instructions and examples, please refer to the [documentation](docs/usage.md).

## Configuration

The clustering algorithm can be configured with the following parameters:

- `n_clusters`: The number of clusters to find
- `linkage`: The linkage criterion to use ('single', 'complete', 'average', 'ward')
- `distance_metric`: The metric to use when calculating distance between instances

