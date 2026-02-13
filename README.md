# Airflow Lab-1: Iris Dataset K-Means Clustering Pipeline

## Overview

This project implements a machine learning pipeline using Apache Airflow to perform K-Means clustering on the Iris dataset. The workflow automates data loading, preprocessing, model training, and evaluation using the elbow method to find optimal cluster counts.

## Features

- Automated ML pipeline orchestration with Apache Airflow
- K-Means clustering on Iris flower dataset (4 features, 150 samples)
- Automatic optimal cluster detection using KneeLocator
- MinMaxScaler for feature normalization
- Docker containerization for easy deployment

## Project Structure
```
airflow_lab1/
├── dags/
│   ├── data/
│   │   ├── file.csv          # Training data (120 samples)
│   │   └── test.csv           # Test data (30 samples)
│   ├── model/
│   │   └── iris_kmeans_model.pkl
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py             # ML functions
│   └── airflow.py             # DAG definition
├── docker-compose.yaml
├── .env
└── generate_iris_data.py
```

## Prerequisites

- Docker Desktop (with 4GB+ RAM allocated)
- Git Bash or terminal
- Python 3.8+ (for dataset generation)

## Setup Instructions

### 1. Generate Dataset
```bash
pip install scikit-learn pandas
python generate_iris_data.py
```

### 2. Configure Environment
```bash
echo "AIRFLOW_UID=$(id -u)" > .env
# If above fails: echo "AIRFLOW_UID=50000" > .env
```

### 3. Initialize and Start Airflow
```bash
docker compose up airflow-init
docker compose up
```

## Usage

1. Open browser: http://localhost:8080
2. Login: username `admin`, password `admin`
3. Find `Airflow_Lab1_Iris` in DAGs list
4. Toggle switch to "On"
5. Click play button and select "Trigger DAG"
6. View results in Graph view > `load_model_task` > Logs

Expected output:
```
Optimal number of clusters: 3
SSE values: [189.34, 78.85, 57.23, ...]
```

## Pipeline Tasks

1. **load_data_task**: Loads Iris CSV and serializes data
2. **data_preprocessing_task**: Applies MinMaxScaler normalization
3. **build_save_model_task**: Trains K-Means for k=1 to 10
4. **load_model_task**: Finds optimal k using elbow method, makes predictions

## Technology Stack

- Apache Airflow 2.7.1
- Docker & Docker Compose
- Python 3.8, scikit-learn, pandas, numpy, kneed
- PostgreSQL (metadata), Redis (message broker)

## Stopping Airflow
```bash
docker compose down
# To remove all data: docker compose down --volumes
```