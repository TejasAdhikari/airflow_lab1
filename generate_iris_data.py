from sklearn.datasets import load_iris
import pandas as pd
import os

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# Create data directory if it doesn't exist
os.makedirs('dags/data', exist_ok=True)

# Save training data (first 120 samples)
df.iloc[:120].to_csv('dags/data/file.csv', index=False)
print("Created dags/data/file.csv with 120 samples")

# Save test data (last 30 samples)
df.iloc[120:].to_csv('dags/data/test.csv', index=False)
print("Created dags/data/test.csv with 30 samples")

print("\nDataset preview:")
print(df.head())