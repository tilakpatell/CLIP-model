import sys
import os
from typing import List, Tuple
import torch
import numpy as np
from models.clip_model import CLIP_MODEL
from models.faiss_model import FAISSIndexer

sys.path.append("/Users/tilakpatel/Desktop/later-data-practicum")

def preprocess_image(clip_model_instance, image):
    """Generate an embedding for a preprocessed image tensor."""
    if image is None:
        raise ValueError("Input image is None")
        
    try:
        with torch.no_grad():
            embedding = clip_model_instance.encode_image(image)
            if embedding is None:
                raise ValueError("CLIP model returned None embedding")
            return embedding
            
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess image: {str(e)}")

def generate_embeddings_using_model(tensor_path: str) -> Tuple[np.ndarray, List[str]]:
    """Generate embeddings for each tensor and store image IDs."""
    # Validate input path
    if not os.path.exists(tensor_path):
        raise FileNotFoundError(f"Tensor path not found: {tensor_path}")
    
    try:
        # Initialize CLIP model
        clip_model_instance = CLIP_MODEL()
        model, preprocess, device = clip_model_instance.load_clip_model()
        
        # Get list of tensor files
        tensor_files = [f for f in os.listdir(tensor_path) if f.endswith('.pt')]
        if not tensor_files:
            raise RuntimeError(f"No .pt files found in {tensor_path}")
        
        print(f"Found {len(tensor_files)} tensor files to process")
        
        # Initialize storage
        vector_embeddings = []
        image_ids = []
        errors = []
        
        # Process each tensor file
        for tensor_file in tensor_files:
            try:
                # Load and validate tensor
                tensor_path_full = os.path.join(tensor_path, tensor_file)
                if not os.path.exists(tensor_path_full):
                    errors.append(f"File not found: {tensor_file}")
                    continue
                
                # Load tensor
                tensor = torch.load(tensor_path_full, map_location=device)
                if tensor is None or tensor.size(0) == 0:
                    errors.append(f"Invalid tensor for {tensor_file}")
                    continue
                
                # Generate embedding
                embedding = preprocess_image(clip_model_instance, tensor)
                if embedding is None:
                    errors.append(f"Failed to generate embedding for {tensor_file}")
                    continue
                
                # Convert to numpy and validate
                embedding_np = embedding.detach().cpu().numpy()
                if embedding_np is None or embedding_np.size == 0:
                    errors.append(f"Invalid embedding for {tensor_file}")
                    continue
                
                # Store valid embedding and ID
                vector_embeddings.append(embedding_np)
                image_ids.append(tensor_file)
                
                # Print progress every 100 files
                if len(vector_embeddings) % 100 == 0:
                    print(f"Processed {len(vector_embeddings)}/{len(tensor_files)} files")
                
            except Exception as e:
                errors.append(f"Error processing {tensor_file}: {str(e)}")
                continue
            
            # Clear GPU memory if needed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Validate results
        if not vector_embeddings:
            if errors:
                print("\nEncountered errors:")
                for error in errors[:10]:  # Show first 10 errors
                    print(f"  {error}")
            raise RuntimeError("No valid embeddings were generated")
        
        # Stack embeddings and validate
        final_embeddings = np.vstack(vector_embeddings)
        if final_embeddings.shape[0] != len(image_ids):
            raise RuntimeError(f"Mismatch between embeddings ({final_embeddings.shape[0]}) and image IDs ({len(image_ids)})")
        
        # Print summary
        print(f"\nEmbedding Generation Summary:")
        print(f"Successfully generated {len(image_ids)} embeddings")
        print(f"Embedding shape: {final_embeddings.shape}")
        print(f"Total errors: {len(errors)}")
        
        if errors:
            print("\nFirst few errors encountered:")
            for error in errors[:5]:
                print(f"  {error}")
        
        return final_embeddings, image_ids
        
    except Exception as e:
        raise RuntimeError(f"Failed to generate embeddings: {str(e)}")

def test_faiss_index(faiss_index_path: str, image_ids_path: str, sample_embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Test FAISS index with a sample embedding."""
    try:
        if sample_embedding is None:
            raise ValueError("Sample embedding cannot be None")
            
        # Initialize FAISS indexer
        faiss_indexer = FAISSIndexer(dim=sample_embedding.shape[1])
        
        # Load index
        faiss_indexer.load_index(faiss_index_path, image_ids_path)
        
        # Search
        results = faiss_indexer.search_with_categories(sample_embedding, k=5)
        
        # Extract distances and indices
        distances = np.array([result['similarity'] for result in results['similar_images']])
        indices = np.array([result['id'] for result in results['similar_images']])
        
        return distances, indices
        
    except Exception as e:
        raise RuntimeError(f"Failed to test FAISS index: {str(e)}")

if __name__ == "__main__":
    try:
        tensor_path = sys.argv[1] if len(sys.argv) > 1 else "/Users/tilakpatel/Desktop/later-data-practicum/normalized_data"
        
        print(f"Generating embeddings from: {tensor_path}")
        embeddings, image_ids = generate_embeddings_using_model(tensor_path)
        
        if len(embeddings) > 0:
            print("\nTesting search functionality:")
            sample_embedding = embeddings[0].reshape(1, -1)
            try:
                distances, indices = test_faiss_index(
                    "faiss_index/index_file.index",
                    "faiss_index/image_ids.pkl",
                    sample_embedding
                )
                print("Test search results:")
                print("Distances:", distances)
                print("Top 5 similar image IDs:", indices)
            except Exception as e:
                print(f"Search test failed: {e}")
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)
