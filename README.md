# HDBSCAN

> A pure Julia implementation of the HDBSCAN clustering algorithm.

This package provides an implementation of **HDBSCAN (Hierarchical Density-Based
Spatial Clustering of Applications with Noise)** based on the reference
implementation available in **scikit-learn 1.8.0**.

[![Julia](https://img.shields.io/badge/Julia-1.12+-9558B2)](https://julialang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success)](https://opensource.org/licenses/MIT)
![Docs](https://img.shields.io/badge/docs-stable-blue)

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Performs DBSCAN over varying epsilon values and integrates 
the result to find a clustering that gives the best stability over epsilon.
This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
and be more robust to parameter selection.

---

## Table of Contents

- [Why HDBSCAN?](#why-hdbscan)
- [User Installation](#user-installation)
- [Quick Start](#quick-start)
- [Core API](#core-functions)
- [Implementation Features](#implementation-features)
- [Implementation Notes](#implementation-notes)
- [Validation](#validation)
- [References](#references)
- [License](#License)
- [Contacts](#Contacts)

---

## Why HDBSCAN?

HDBSCAN returns a good clustering with little or no parameter tuning. It's 
ideal for exploratory data analysis. It's a fast and robust
algorithm that reliably returns meaningful clusters (if there are any).

---

## User Installation

Install from GitHub
```julia
using Pkg
Pkg.add(url="https://github.com/felice-liu/HDBSCAN.jl")

```
or install the registered package directly

```julia
using Pkg
Pkg.add("HDBSCAN")

```
## Quick Start

A simple clustering example is provided below.

```julia
import Pkg
Pkg.add("MLJ")
import HDBSCAN
import MLJ

X, y = MLJ.make_blobs(1000, 2, as_table=false)

# min_cluster_size = 15, min_sample_size = 5
model = HDBSCAN.Hdbscan(15, 5; metric="euclidean")

HDBSCAN.fit!(model, X)

labels = HDBSCAN.labels(model)
probabilities = HDBSCAN.probabilities(model)

println(labels)
println(probabilities)
```
---

## Core API

- `Hdbscan`
- `fit!`
- `fit_predict`
- `labels`
- `probabilities`
- `centroids`
- `medoids`
- `single_linkage_tree`
- `nclusters`

---

## Implementation Features

- Pure Julia implementation of HDBSCAN clustering

- Distance metrics integrated via Distances.jl

- Brute-force, KDTree and BallTree algorithms

- Excess of mass (EOM) and Leaf selection

- Compute cluster centroids and medoids

---

## Implementation notes

Minor numerical differences may occur due to floating-point arithmetic for the
Euclidean and WEuclidean metrics using BLAS3 matrix computation of Distances.jl.
The library is currently using sortperm instead of numpy argsort. Both of these
features make minor differences on the clustering results on tie-breaking decisions
on overlapping points. Support for "precomputed" metric is currently limited.

---

## Speed Benchmark

The benchmark compares this implementation against the official scikit-learn implementation.
Ratios greater than 1 indicate that Julia is faster than Python.

|   Dataset         |    Python fit time    |    Julia fit time    |    Python/Julia    |
|-------------------|-----------------------|----------------------|--------------------|
|   Aniso           |   1,04E-02            |   7,16E-03           |    **1.46x**       |
|   Blobs           |   1,06E-02            |   7,22E-03           |    **1.47x**       |
|   Circles         |   1,17E-02            |   7,52E-03           |    **1,56x**       |
|   Moons           |   1,07E-02            |   7,01E-03           |    **1,52x**       |
|   NoStructure     |   1,05E-02            |   7,25E-03           |    **1,45x**       |
|   Varied          |   1,03E-02            |   7,10E-03           |    **1.45x**       |
|   CardiacArrest   |   1,12E-02            |   7,13E-03           |    **1,57x**       |
|   HeartFailure    |   1,43E-02            |   9,07E-03           |    **1,57x**       |
|   NeuroBlastoma   |   4,22E-03            |   1,86E-03           |    **2,27x**       |
|   Sepsis          |   7,56E-02            |   5,68E-02           |    **1,33x**       |
|   Type1Diabetes   |   2,79E-03            |   7,51E-04           |    **3,71x**       |

---

## Validation

This implementation has been extensively validated against the official
scikit-learn HDBSCAN implementation.

Validation includes:

- Synthetic benchmark datasets
- Real-world datasets
- Comparison of:
    - cluster labels
    - membership probabilities
    - DBCV scores


---

## References

> Ricardo JGB Campello, et al. "Hierarchical density estimates for data clustering, visualization,
and outlier detection." ACM Transactions on Knowledge Discovery from Data (TKDD) 10.1 (2015): 1-51.
[https://doi.org/10.1145/2733381](https://doi.org/10.1145/2733381)

---
## License

MIT (c) Liu Felice

---

## Contacts
For any enquire, please write to Liu Felice at [f.liu3(AT)campus.unimib.it](mailto:f.liu3@campus.unimib.it)