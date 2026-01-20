# Cancer Classification using KNN (R)

This project implements and evaluates a **K-Nearest Neighbors (KNN)** classifier to diagnose cancer as either malignant or benign based on cell nucleus characteristics.

## Project Overview

The goal is to build a predictive model that can accurately classify patients based on digitized images of a fine needle aspirate (FNA) of a mass. 

### Data Description
The dataset contains information regarding:
1. **ID number**: Patient identifier.
2. **Diagnosis**: Target variable (**M** = malignant, **B** = benign).

### Feature Details
Ten real-valued features were computed for each cell nucleus:
* **Radius**: Mean of distances from center to points on the perimeter.
* **Texture**: Standard deviation of gray-scale values.
* **Perimeter**: The distance around the nucleus.
* **Area**: The total surface area of the nucleus.
* **Smoothness**: Local variation in radius lengths.
* **Compactness**: $perimeter^2 / area - 1.0$.
* **Concavity**: Severity of concave portions of the contour.
* **Concave points**: Number of concave portions of the contour.
* **Symmetry**: Balance of the nucleus shape.
* **Fractal dimension**: "Coastline approximation" - 1.

The **mean**, **standard error (SE)**, and **"worst"** (mean of the three largest values) of these features were computed for each image, resulting in **30 features** total. For instance:
* Field 3 is **Mean Radius**.
* Field 13 is **Radius SE**.
* Field 23 is **Worst Radius**.

## Tech Stack
* **Language**: R
* **Libraries**: 
  * `class`, `caret`: For KNN implementation.
  * `tidyverse` (ggplot2, dplyr): For data manipulation and visualization.
  
