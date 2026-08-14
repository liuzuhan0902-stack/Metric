import numpy as np
import json
import os

def generate_synthetic_data(num_samples=100, dimension=128, output_dir="data"):
    """
    Generates synthetic dataset for MetricDB experimentation.
    
    Args:
        num_samples (int): Number of entities to generate.
        dimension (int): Dimensionality of the vector embeddings.
        output_dir (str): Directory where the generated files will be saved.
        
    This function creates:
    - vectors.npy: A NumPy array of shape (num_samples, dimension).
    - metadata.json: A list of dictionaries containing 'id', 'category', 'price', etc.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating {num_samples} samples with dimension {dimension}...")
    
    # Generate random vectors (simulating embeddings from some model)
    vectors = np.random.rand(num_samples, dimension).astype(np.float32)
    np.save(os.path.join(output_dir, "vectors.npy"), vectors)
    
    # Generate structured metadata for relational filtering
    metadata = []
    categories = ["electronics", "clothing", "home", "beauty", "toys"]
    for i in range(num_samples):
        metadata.append({
            "id": f"item_{i}",
            "category": np.random.choice(categories),
            "price": round(np.random.uniform(10, 1000), 2),
            "timestamp": "2026-01-19T13:30:00"
        })
        
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Dataset generated successfully in '{output_dir}/'.")

if __name__ == "__main__":
    # Example usage: generate a small dataset for quick testing
    generate_synthetic_data(num_samples=100, dimension=128)
