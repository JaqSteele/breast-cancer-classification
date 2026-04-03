# Breast Cancer Classification

## Overview
This project applies machine learning models to classify breast cancer cases as malignant or benign using the Wisconsin Breast Cancer dataset.

The goal is to demonstrate a clean end-to-end machine learning pipeline, including data preprocessing, model training, evaluation, and result visualisation.

## Models Used
- Logistic Regression
- Random Forest

## Results

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | ~95.6%  |
| Random Forest       | ~97.1%  |

The Random Forest model achieved the highest performance, demonstrating the benefit of ensemble methods on structured medical data.

## Features
- Data cleaning (handling missing values)
- Feature scaling
- Model comparison
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Results saved as CSV
- Visualisation of model performance

## Repository Structure
- `code/` – model training script
- `results/` – saved metrics and plots
- `requirements.txt` – dependencies

## Purpose
This project demonstrates the application of machine learning to a real-world healthcare dataset, with a focus on building a clean and reproducible pipeline.

## Note
This is a simplified classification project intended for demonstration and learning purposes.
