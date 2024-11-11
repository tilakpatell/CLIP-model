import faiss
import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict

class FAISSIndexer:
    """FAISS Indexer supporting multi-category assignments and similarity tracking."""
    
    def __init__(self, dim: int = 512, 
                 similarity_threshold: float = 0.75,
                 max_categories_per_image: int = 3):
        """Initialize FAISS indexer with given parameters."""
        if dim <= 0:
            raise ValueError("Dimension must be positive")
        if similarity_threshold <= 0 or similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        self.index = faiss.IndexFlatL2(dim)
        self.image_ids = []
        self.similarity_scores = {}
        self.category_metadata = {
            'centroids': {},         # {category_id: np.array}
            'members': {},           # {category_id: [image_ids]}
            'hierarchy': {},         # {category_id: {'parent': id, 'children': [ids]}}
            'relationships': {}      # {category_id: {related_id: similarity}}
        }
        self.similarity_threshold = similarity_threshold
        self.max_categories = max_categories_per_image

    def add_embeddings_with_categories(self, embeddings: np.ndarray, image_ids: List[str]) -> Dict:
        """Add embeddings with category assignments."""
        # Input validation
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings cannot be None or empty")
        if image_ids is None or len(image_ids) == 0:
            raise ValueError("Image IDs cannot be None or empty")
        if len(embeddings) != len(image_ids):
            raise ValueError(f"Number of embeddings ({len(embeddings)}) must match number of image IDs ({len(image_ids)})")

        try:
            # Debug info
            print(f"Embeddings shape before normalization: {embeddings.shape}")
            print(f"Embeddings min: {embeddings.min()}, max: {embeddings.max()}")
            print(f"Embeddings mean: {embeddings.mean()}, std: {embeddings.std()}")
            
            # Manual normalization
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Prevent division by zero
            normalized = embeddings / norms
            
            print(f"Processing {len(image_ids)} embeddings")
            results = {}
            
            for idx, (embedding, image_id) in enumerate(zip(normalized, image_ids)):
                if embedding is None or image_id is None:
                    print(f"Skipping invalid data at index {idx}")
                    continue

                # Reshape embedding for FAISS
                embedding_vec = embedding.reshape(1, -1)
                if np.any(np.isnan(embedding_vec)):
                    print(f"Warning: NaN values in embedding for {image_id}")
                    continue
                
                # Initialize result structure
                results[image_id] = {
                    'main_category': None,
                    'all_categories': {},
                    'new_categories_created': []
                }
                
                # Handle first embedding case
                if self.index.ntotal == 0:
                    category_id = self._generate_category_id()
                    self.create_new_category(embedding_vec, image_id)
                    results[image_id]['main_category'] = category_id
                    results[image_id]['all_categories'][category_id] = 1.0
                    self.similarity_scores[image_id] = {category_id: 1.0}
                    self.index.add(embedding_vec)
                    self.image_ids.append(image_id)
                    continue
                
                # Find similar vectors
                k = min(5, self.index.ntotal)
                D, I = self.index.search(embedding_vec, k)
                similarities = 1 - (D[0] / (2 * self.index.ntotal))
                
                # Find candidate categories
                candidate_categories = {}
                for neighbor_idx, similarity in zip(I[0], similarities):
                    neighbor_id = self.image_ids[neighbor_idx]
                    if neighbor_id in self.similarity_scores:
                        for cat_id, score in self.similarity_scores[neighbor_id].items():
                            if cat_id not in candidate_categories:
                                candidate_categories[cat_id] = []
                            candidate_categories[cat_id].append(similarity * score)
                
                # Calculate category similarities
                category_similarities = {}
                for cat_id, scores in candidate_categories.items():
                    if scores:  # Check if scores list is not empty
                        category_similarities[cat_id] = np.mean(scores)
                
                # Sort categories by similarity
                sorted_categories = sorted(
                    category_similarities.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Assign categories
                assigned_categories = []
                for cat_id, similarity in sorted_categories:
                    if similarity >= self.similarity_threshold:
                        if len(assigned_categories) < self.max_categories:
                            assigned_categories.append((cat_id, similarity))
                            self.update_category_metadata(cat_id, embedding_vec, image_id)
                
                # Create new category if needed
                if not assigned_categories:
                    new_cat_id = self.create_new_category(embedding_vec, image_id)
                    assigned_categories.append((new_cat_id, 1.0))
                    results[image_id]['new_categories_created'].append(new_cat_id)
                
                # Update results
                if assigned_categories:
                    results[image_id]['primary_category'] = assigned_categories[0][0]
                    results[image_id]['all_categories'] = {
                        cat_id: sim for cat_id, sim in assigned_categories
                    }
                
                # Update tracking
                self.similarity_scores[image_id] = results[image_id]['all_categories']
                self.index.add(embedding_vec)
                self.image_ids.append(image_id)
                
                # Print progress every 50 images
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(image_ids)} images")
            
            print(f"Completed processing all {len(image_ids)} images")
            return results
            
        except Exception as e:
            print(f"Error details - embeddings shape: {embeddings.shape if embeddings is not None else 'None'}")
            print(f"Number of image IDs: {len(image_ids)}")
            raise RuntimeError(f"Failed to add embeddings: {str(e)}")

    def update_category_metadata(self, category_id: str, new_embedding: np.ndarray, image_id: str):
        """Update category metadata with new embedding."""
        try:
            # First, ensure the category exists
            if category_id not in self.category_metadata['centroids']:
                raise ValueError(f"Category {category_id} not found in centroids")
            
            # Get current members and validate
            members = self.category_metadata['members'].get(category_id, [])
            if not isinstance(members, list):
                members = []
            
            # Get current centroid and validate
            old_centroid = self.category_metadata['centroids'].get(category_id)
            if old_centroid is None:
                new_centroid = new_embedding
            else:
                try:
                    new_centroid = (old_centroid * len(members) + new_embedding) / (len(members) + 1)
                except ValueError as e:
                    print(f"Error calculating centroid: {str(e)}")
                    print(f"old_centroid shape: {old_centroid.shape if old_centroid is not None else None}")
                    print(f"new_embedding shape: {new_embedding.shape if new_embedding is not None else None}")
                    print(f"members length: {len(members)}")
                    raise

            # Update metadata
            self.category_metadata['centroids'][category_id] = new_centroid
            if image_id not in members:
                self.category_metadata['members'][category_id] = members + [image_id]

            # Initialize relationships if needed
            if category_id not in self.category_metadata['relationships']:
                self.category_metadata['relationships'][category_id] = {}

            # Update relationships
            for other_id in self.category_metadata['centroids']:
                if other_id != category_id:
                    try:
                        similarity = np.dot(
                            new_centroid.flatten(),
                            self.category_metadata['centroids'][other_id].flatten()
                        )
                        
                        if other_id not in self.category_metadata['relationships']:
                            self.category_metadata['relationships'][other_id] = {}
                            
                        self.category_metadata['relationships'][category_id][other_id] = similarity
                        self.category_metadata['relationships'][other_id][category_id] = similarity
                        
                    except Exception as e:
                        print(f"Error updating relationship between {category_id} and {other_id}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Category ID: {category_id}")
            print(f"Image ID: {image_id}")
            print(f"New embedding shape: {new_embedding.shape if new_embedding is not None else None}")
            print(f"Current category metadata state:")
            print(f"- Centroids keys: {list(self.category_metadata['centroids'].keys())}")
            print(f"- Members keys: {list(self.category_metadata['members'].keys())}")
            raise RuntimeError(f"Failed to update category metadata: {str(e)}")

    def create_new_category(self, embedding_vec: np.ndarray, image_id: str, parent_id: Optional[str] = None) -> str:
      """Create a new category with initial embedding."""
      try:
          # Generate unique category ID
          existing_ids = set(self.category_metadata['centroids'].keys())
          category_id = "cat_0"
          counter = 0
          while category_id in existing_ids:
              counter += 1
              category_id = f"cat_{counter}"
          
          print(f"Creating new category with ID: {category_id}")
          
          # Ensure embedding is 2D
          if len(embedding_vec.shape) == 1:
              embedding_vec = embedding_vec.reshape(1, -1)
          
          if np.any(np.isnan(embedding_vec)):
              raise ValueError("Embedding contains NaN values")
              
          # Initialize category metadata
          self.category_metadata['centroids'][category_id] = embedding_vec.copy()
          self.category_metadata['members'][category_id] = [image_id]  # Initialize with single member
          self.category_metadata['hierarchy'][category_id] = {
              'parent': parent_id,
              'children': []
          }
          self.category_metadata['relationships'][category_id] = {}
          
          # Initialize similarity score
          if image_id not in self.similarity_scores:
              self.similarity_scores[image_id] = {}
          self.similarity_scores[image_id][category_id] = 1.0
          
          # Update parent if exists
          if parent_id and parent_id in self.category_metadata['hierarchy']:
              self.category_metadata['hierarchy'][parent_id]['children'].append(category_id)
              
          return category_id
          
      except Exception as e:
          print(f"Failed to create category with embedding shape: {embedding_vec.shape if embedding_vec is not None else None}")
          raise RuntimeError(f"Failed to create category: {str(e)}")

    def _generate_category_id(self) -> str:
        """Generate unique category ID."""
        next_id = len(self.category_metadata['centroids'])
        return f"cat_{next_id}"

    def search_with_categories(self, query_embedding: np.ndarray, k: int = 5) -> Dict:
        """Search for similar images and their categories."""
        try:
            # Normalize query embedding
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            query_embedding = query_embedding.reshape(1, -1)
            D, I = self.index.search(query_embedding, k)
            similarities = 1 - (D[0] / (2 * self.index.ntotal))

            results = {
                'similar_images': [],
                'suggested_categories': {}
            }

            for idx, (neighbor_idx, similarity) in enumerate(zip(I[0], similarities)):
                image_id = self.image_ids[neighbor_idx]
                image_categories = self.similarity_scores.get(image_id, {})
                results['similar_images'].append({
                    'id': image_id,
                    'similarity': float(similarity),
                    'categories': image_categories
                })

                for cat_id, cat_score in image_categories.items():
                    if cat_id not in results['suggested_categories']:
                        results['suggested_categories'][cat_id] = []
                    results['suggested_categories'][cat_id].append(similarity * cat_score)

            results['suggested_categories'] = {
                cat_id: float(np.mean(scores))
                for cat_id, scores in results['suggested_categories'].items()
            }

            return results

        except Exception as e:
            raise RuntimeError(f"Search failed: {str(e)}")

    def save_index(self, path: str):
        """Save index and metadata to disk."""
        try:
            data = {
                'index_data': faiss.serialize_index(self.index),
                'image_ids': self.image_ids,
                'similarity_scores': self.similarity_scores,
                'category_metadata': self.category_metadata
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save index: {str(e)}")

    def load_index(self, path: str):
        """Load index and metadata from disk."""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.index = faiss.deserialize_index(data['index_data'])
            self.image_ids = data['image_ids']
            self.similarity_scores = data['similarity_scores']
            self.category_metadata = data['category_metadata']
        except Exception as e:
            raise RuntimeError(f"Failed to load index: {str(e)}")
