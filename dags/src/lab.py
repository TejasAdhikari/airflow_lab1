import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from kneed import KneeLocator
import os
import base64

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("Loading Iris dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    
    # Serialize DataFrame to bytes, then encode as base64 string
    # this is because XCom can't directly pass DataFrames, but can pass strings
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # Decode base64 -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    # Drop any rows with missing values
    df = df.dropna()
    
    # Select features for clustering (all 4 iris features)
    clustering_data = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    
    # MinMaxScaler: scales all features to [0, 1] range
    # K-Means is distance-based, so features need to be on same scale
    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)
    
    # Serialize preprocessed data back to base64 string
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")

def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # Decode base64 -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)
    
    # K-Means configuration
    kmeans_kwargs = {
        "init": "random",      # Random initialization
        "n_init": 10,          # Run 10 times with different initializations
        "max_iter": 300,       # Max iterations per run
        "random_state": 42     # For reproducibility
    }
    
    # Try different numbers of clusters and calculate SSE (inertia)
    sse = []
    for k in range(1, 11):  # Test k=1 to k=10 (adjust range as needed)
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)  # SSE for this k value
    
    # Save the last fitted model (k=10 in this case)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)
    
    print(f"Model saved to {output_path}")
    return sse  # List is JSON-safe, can be passed through XCom

def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report optimal k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # Load the saved model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))
    
    # Use KneeLocator to automatically find the elbow point
    # Instead of manually looking at SSE plot, this finds the optimal k
    kl = KneeLocator(
        range(1, len(sse) + 1),   # k values tested
        sse,                      # SSE values
        curve="convex",           # SSE curve is convex
        direction="decreasing"    # SSE decreases as k increases
    )
    
    print(f"Optimal number of clusters: {kl.elbow}")
    print(f"SSE values: {sse}")
    
    # Make predictions on test data
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    predictions = loaded_model.predict(test_df)
    
    print(f"First test sample predicted cluster: {predictions[0]}")
    
    # Return first prediction as JSON-safe integer
    try:
        return int(predictions[0])
    except Exception:
        return predictions[0].item() if hasattr(predictions[0], "item") else predictions[0]