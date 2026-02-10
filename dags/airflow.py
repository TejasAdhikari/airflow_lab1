# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow
from airflow import configuration as conf

# Enable pickle support for XCom
# XCom normally only handles JSON, but we need to pass complex Python objects
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
# These apply to all tasks unless overridden
default_args = {
    'owner': 'your_name',  # Who owns this DAG (for organization)
    'start_date': datetime(2026, 2, 9),  # When this DAG becomes active
    'retries': 0,  # How many times to retry failed tasks (0 = no retries)
    'retry_delay': timedelta(minutes=5),  # Wait 5 minutes between retries
}

# Create a DAG instance
dag = DAG(
    'Airflow_Lab1_Iris',  # DAG name - DIFFERENT from repo (shows up in Airflow UI)
    default_args=default_args,
    description='Iris Dataset K-Means Clustering Pipeline',
    schedule_interval=None,  # None = manual trigger only (not scheduled)
    catchup=False,  # False = don't run for past dates when first enabled
)

# Task 1: Load Iris data from CSV
# PythonOperator executes a Python function as an Airflow task
load_data_task = PythonOperator(
    task_id='load_data_task',  # Unique identifier for this task
    python_callable=load_data,  # The function to execute
    dag=dag,  # Which DAG this task belongs to
)

# Task 2: Preprocess the data (MinMaxScaler)
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],  # Pass output from load_data_task as input
    dag=dag,
)

# Task 3: Build and save the K-Means model
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=build_save_model,
    op_args=[data_preprocessing_task.output, "iris_kmeans_model.pkl"],
    # Pass preprocessed data and the file path to save the model
    provide_context=True,  # Gives access to Airflow context variables
    dag=dag,
)

# Task 4: Load model and find optimal cluster count using elbow method
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=load_model_elbow,
    op_args=["iris_kmeans_model.pkl", build_save_model_task.output],
    # Pass model path and SSE values from previous task
    dag=dag,
)

# Set task dependencies (execution order)
# This creates the pipeline
load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task