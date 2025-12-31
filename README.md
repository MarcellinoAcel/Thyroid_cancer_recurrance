# Thyroid Cancer Recurrence Prediction

This repository contains a machine learning project for predicting **thyroid cancer recurrence** using clinical data.  
The objective of this project is to build and evaluate classification models that can estimate whether thyroid cancer will recur based on patient clinical features.

---

## Project Overview

Thyroid cancer is one of the most common endocrine malignancies. Early prediction of recurrence is important for treatment planning and long-term patient monitoring.

This project implements a complete machine learning workflow including:

- Data loading and exploration
- Data preprocessing
- Model training and evaluation
- Result visualization

All experiments are implemented in **Python using Jupyter Notebooks**.

---

## Dataset Source

The dataset used in this project was obtained from Kaggle:

Thyroid Cancer Dataset  
https://www.kaggle.com/datasets/khwaishsaxena/thyroid-cancer-dataset

The dataset contains clinical and demographic information related to thyroid cancer patients, along with a recurrence label.

---

## Repository Structure

Thyroid_cancer_recurrance/
├── Thyroid Cancer.pptx
├── label_ml/
├── thyroid_data.csv
├── thyroid_recurrance.ipynb
└── README.md

---

## Dataset Description

The dataset includes clinical and demographic features such as:

- Age
- Gender
- Medical history
- Clinical examination results
- Thyroid cancer recurrence label

The full description of features and labels can be found inside the notebook and dataset file.

---

## Key Features

- Real-world clinical dataset
- Data preprocessing and feature engineering
- Machine learning classification models
- Model evaluation using standard metrics
- Visualization of results

---

## Requirements

Make sure you have the following installed:

- Python 3.8+
- Jupyter Notebook or Jupyter Lab
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## Installation

Clone the repository:

git clone https://github.com/MarcellinoAcel/Thyroid_cancer_recurrance.git  
cd Thyroid_cancer_recurrance

(Optional) Create a virtual environment:

python -m venv venv  
source venv/bin/activate   (Linux / macOS)  
venv\Scripts\activate      (Windows)

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn jupyter

---

## Running the Notebook

Start Jupyter Notebook:

jupyter notebook

Open and run:

thyroid_recurrance.ipynb  
This notebook contains the full pipeline: data preprocessing, model training, evaluation, and visualization.

---

## Methodology

1. Data Preprocessing  
   - Handle missing values  
   - Encode categorical variables  
   - Feature scaling  

2. Model Training  
   - Train-test split  
   - Train classification models  

3. Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - Confusion matrix  

---

## Results

Model performance results and visualizations are shown directly in the notebook.  
Different algorithms can be compared to identify the best-performing model.

---

## Future Improvements

- Hyperparameter tuning
- Feature selection
- Advanced models (XGBoost, Neural Networks)
- Model deployment for real-world usage

---

## Author

Bernard Marcellino Sitio  
AI & Machine Learning Enthusiast  
GitHub: https://github.com/MarcellinoAcel
