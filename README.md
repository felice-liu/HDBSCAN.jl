# HDBSCAN

> Julia implementation of HDBSCAN clustering algorithm

This library is a implementation of HDBSCAN based on the public library Scikit
Learn 1.8.0

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Julia-%3E%3D%201.12.6-FF3131)](https://julialang.org/)

=======
HDBSCAN
=======

HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications
with Noise. Performs DBSCAN over varying epsilon values and integrates 
the result to find a clustering that gives the best stability over epsilon.
This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN),
and be more robust to parameter selection.

---

## Table of Contents

- [Why HDBSCAN?](#why-hdbscan)
- [Implementation Features](#implementation-features)
- [User Installation](#user-installation)
- [Quick Start](#quick-start)
- [Core Functions](#core-functions)
- [Implementation Notes](#implementation-notes)
- [Validation](#validation)
- [References](#references)

---

## Why HDBSCAN?

HDBSCAN returns a good clustering with little or no parameter tuning. It's 
ideal for exploratory data analysis. It's a fast and robust
algorithm that reliably returns meaningful clusters (if there are any).

---

## Implementation Features

- Pure Julia implementation of HDBSCAN clustering

- Distance metrics integrated via Distances.jl

- Brute-force, KDTree and BallTree algorithms

- Excess of mass (EOM) and Leaf selection

- Compute cluster centroids and medoids

---

## User Installation

Install from GitHub

```julia
using Pkg
Pkg.add(url="https://github.com/felice-liu/HDBSCAN-Julia-library/blob/master/src/hdbscan.jl")

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
Pkg.add("SyntheticDatasets")
import HDBSCAN
import SyntheticDatasets

blobs = SyntheticDatasets.make_blobs(
    n_samples = 1000, 
    n_features = 2,
    centers = [-1 1; -0.5 0.5], 
    cluster_std = 0.25,
    center_box = (-2.0, 2.0), 
    shuffle = true,
    random_state = nothing)

# min_cluster_size = 15, min_sample_size = 5
model = Hdbscan(15, 5; metric="euclidean")

fit!(model, blobs)

labels = labels(model)
probabilities = probabilities(model)

println(labels)
println(probabilities)
```
---

## Core Functions

Main API

Hdbscan, fit!, fit_predict, labels, probabilities, centroids, medoids, single_linkage_tree
nclusters

Documentation for every public function is available from the Julia REPL!

```text
help?> Hdbscan
help?> fit!
help?> fit_predict
```

---

## Implementation notes

Minor numerical differences may occur due to floating-point arithmetic. This can make
a difference on the clustering result on very close points.
Support for "precomputed" metric is currently limited.

---

## Speed Benchmark

Each hyperparameter combination's fit time was obtained by averaging 20 fit times

|   Dataset         |    Python fit time    |    Julia fit time    |    Python/Julia    |
|-------------------|-----------------------|----------------------|--------------------|
|   Aniso           |   8,16E-03            |   5,67E-01           |    0.01x           |
|-------------------|-----------------------|----------------------|--------------------|
|   Blobs           |   8,39E-03            |   7,92E-03           |    **1.06x**       |
|-------------------|-----------------------|----------------------|--------------------|
|   Circles         |   9,06E-03            |   8,74E-03           |    **1.03x**       |
|-------------------|-----------------------|----------------------|--------------------|
|   Moons           |   8,20E-03            |   1,17E-02           |    0.7X            |
|-------------------|-----------------------|----------------------|--------------------|
|   NoStructure     |   8,23E-03            |   7,28E-03           |    **1.13x**       |
|-------------------|-----------------------|----------------------|--------------------|
|   Varied          |   8,10E-03            |   7,13E-03           |    **1.13x**       |
|-------------------|-----------------------|----------------------|--------------------|
|   CardiacArrest   |   9,61E-03            |   4,45E-02           |    0.21x           |
|-------------------|-----------------------|----------------------|--------------------|
|   HeartFailure    |   1,16E-02            |   6,55E-02           |    0.18x           |
|-------------------|-----------------------|----------------------|--------------------|
|   NeuroBlastoma   |   3,40E-03            |   4,74E-02           |    0.07x           |
|-------------------|-----------------------|----------------------|--------------------|
|   Sepsis          |   6,36E-02            |   9,39E-02           |    0.68x           |
|-------------------|-----------------------|----------------------|--------------------|
|   Type1Diabetes   |   2,24E-03            |   4,80E-02           |    0,04x           |

Highlighted values (ratio > 1) shows Julia is faster than Python.

---

## Validation

The implementation has been validated against the official scikit-learn implementation.

Validation includes:

- Synthetic benchmark datasets
- Real-world datasets
- Comparison of:
    - cluster labels
    - membership probabilities
    - DBCV scores

---

## References

Based on the papers:

    McInnes L, Healy J. *Accelerated Hierarchical Density Based Clustering* 
    In: 2017 IEEE International Conference on Data Mining Workshops (ICDMW), IEEE, pp 33-42.
    2017 `[pdf] <http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8215642>`_

    R. Campello, D. Moulavi, and J. Sander, *Density-Based Clustering Based on
    Hierarchical Density Estimates*
    In: Advances in Knowledge Discovery and Data Mining, Springer, pp 160-172.
    2013 
    
Notebooks `comparing HDBSCAN to other clustering algorithms <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/Comparing%20Clustering%20Algorithms.ipynb>`_, explaining `how HDBSCAN works <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/How%20HDBSCAN%20Works.ipynb>`_ and `comparing performance with other python clustering implementations <http://nbviewer.jupyter.org/github/scikit-learn-contrib/hdbscan/blob/master/notebooks/
