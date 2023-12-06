# Tiling and PCA Strategy for Clustering-based High-Dimensional Gaussian Filtering
This page provides the code, binary, and image dataset of "Tiling and PCA Strategy for Clustering-based High-Dimensional Gaussian Filtering".

# Paper
Sou Oishi and Norishige Fukushima, "Tiling and PCA strategy for Clustering-based High-Dimensional Gaussian Filtering", SN Computer Science, vol. 5 article number 40, 2023.

## bibtex
```
@article{oishi2023tiling,
    author  = {Sou Oishi and Norishige Fukushima},
    title   = {Tiling and PCA Strategy for Clustering-Based High-Dimensional Gaussian Filtering},
    journal = {SN Computer Science},
    volume  = {5},
    number  = {40},
    pages   = {1--20},
    year    = {2023},
    doi     = {10.1007/s42979-023-02319-6},
}
```

## abstract
**Purpose:**
Edge-preserving filtering is an essential tool for image processing applications and has various types of filtering.
High-dimensional Gaussian filtering (HDGF) supports a wide range of edge-preserving filtering.
This paper approximates HDGF by clustering with Nystr\"om approximation, tiling, and principal component analysis (PCA) to accelerate HDGF.
Also, we compare it with the conventional HDGF approximations and clarify its effective range.

**Methods:**
We accelerate HDGF by clustering-based constant-time algorithm, which has $O(K)$ order for convolution, where $K$ is the number of clusters.
First, we perform PCA for dimensionality reduction and then cluster signals with k-means++.
HDGF is decomposed to Gaussian filtering by approximate eigenvalue decomposition of Nystr\"om approximation using the clusters.
The Gaussian filtering is performed in a constant-time algorithm.
The process is further accelerated by the tiling strategy cooperating with PCA.

**Results:** 
In our experimental results, we compared three approximated HDGFs: clustering-based HDGF, permutohedral lattice, and Gaussian KD-tree.
Also, we evaluated six types of high dimensional signals: RGB, RGB-IR, RGB-D, flash/no-flash, hyperspectral image, and non-local means.
The proposed clustering-based HDGF was effective for low/middle-dimensional cases: RGB, RGB-IR, RGB-D, flash/no-flash, and hyperspectral images.
Also, tiling with PCA strategy is effective for the conventional permutohedral lattice and Gaussian KD-tree.

In the approximation of HDGF, the clustering-based HDGF is the better solution for low/middle-dimensional signals.
For the higher-dimensional case of non-local means filtering, the conventional HDGF of the permutohedral lattice with the proposed PCA tiling is effective.

# Code
The Code is available at [GitHub](https://github.com/norishigefukushima/TilingPCA4CHDGF).
New, we are making a command line tool for this project.

# Dataset
The image dataset for our paper can be downloaded.
[link](https://github.com/norishigefukushima/TilingPCA4CHDGF/tree/main/clusteringBasedBilateralFilterTest/img)



