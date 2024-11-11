import faiss
import numpy as np
from typing import List, Dict, Tuple
from models.faiss_model import FAISSIndexer
import shutil
import os
from models.vlm_model import VLMProcessor
from PIL import Image
from sklearn.cluster import AgglomerativeClustering



class CategoryManager:
  def __init__(self, faiss_indexer: FAISSIndexer, min_samples_category: int = 3, max_category_radius: float = 0.3):
    if not isinstance(faiss_indexer, FAISSIndexer):
        raise ValueError("faiss_indexer must be an instance of FAISSIndexer")
        
    if min_samples_category < 2:
        raise ValueError("min_samples_category must be at least 2")
    if max_category_radius <= 0 or max_category_radius >= 1:
        raise ValueError("max_category_radius must be between 0 and 1")
    
    self.faiss_indexer = faiss_indexer
    self.min_samples = min_samples_category
    self.max_radius = max_category_radius 
    
    self.category_labels = {}
    self.category_statistics = {}
    self.vlm_processor = VLMProcessor()
    self.image_descriptions = {}
 
  def _identify_subcategories(self, embeddings: np.ndarray, image_ids: List[str], 
                          main_category_id: str) -> Dict[str, List[str]]:
    """Identify subcategories within a main category using hierarchical clustering."""
    try:
        if len(embeddings) < 2 * self.min_samples:
            return {}
        
        max_subcategories = min(
            int(np.sqrt(len(embeddings))),
            max(2, len(embeddings) // (2 * self.min_samples))
        )
        
        clustering = AgglomerativeClustering(
            n_clusters=max_subcategories,
            metric='euclidean',
            linkage='ward'
        )
        
        subcategory_assignments = clustering.fit_predict(embeddings)
        
        subcategories = {}
        for i in range(max_subcategories):
            mask = subcategory_assignments == i
            if np.sum(mask) >= self.min_samples:
                subcat_embeddings = embeddings[mask]
                subcat_ids = [image_ids[j] for j, m in enumerate(mask) if m]
                
                subcat_id = self.faiss_indexer.create_new_category(
                    embedding_vec=np.mean(subcat_embeddings, axis=0).reshape(1, -1),
                    image_id=subcat_ids[0],
                    parent_id=main_category_id
                )
                
                self.faiss_indexer.category_metadata['hierarchy'][main_category_id]['children'].append(subcat_id)
                
                self.faiss_indexer.category_metadata['members'][subcat_id] = subcat_ids
                subcategories[subcat_id] = subcat_ids
                
                self.category_statistics[subcat_id] = {
                    'diversity': self._calculate_diversity(subcat_embeddings),
                    'quality': self._calculate_quality(
                        subcat_embeddings,
                        np.mean(subcat_embeddings, axis=0)
                    )
                }
        
        return subcategories
        
    except Exception as e:
        print(f"Failed to identify subcategories: {str(e)}")
        return {}
      
  def _determine_clustering_parameters(self, embeddings: np.ndarray) -> dict:
    """Dynamically determine clustering parameters based on data."""
    n_samples = len(embeddings)
    
    # Calculate pairwise distances for a sample of points
    sample_size = min(1000, n_samples)
    indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_embeddings = embeddings[indices]
    
    distances = np.linalg.norm(
        sample_embeddings[:, np.newaxis] - sample_embeddings, 
        axis=2
    )
    
    # Calculate statistics
    mean_dist = np.mean(distances[distances > 0])
    std_dist = np.std(distances[distances > 0])
    
    # Dynamic parameters
    params = {
        'n_clusters': self._calculate_optimal_clusters(n_samples),
        'similarity_threshold': max(0.5, 1 - (mean_dist / (2 * std_dist))),
        'min_samples': max(2, int(np.log2(n_samples))),
        'max_radius': mean_dist + std_dist
    }
    
    print("\nDynamic clustering parameters:")
    print(f"Number of clusters: {params['n_clusters']}")
    print(f"Similarity threshold: {params['similarity_threshold']:.3f}")
    print(f"Minimum samples per cluster: {params['min_samples']}")
    print(f"Maximum cluster radius: {params['max_radius']:.3f}")
    
    return params
   
  
    
  def process_batch(self, embeddings: np.ndarray, image_ids: List[str], 
                 image_folder: str) -> Dict:
    """Process images with dynamic clustering."""
    try:
        print("\nAnalyzing dataset characteristics...")
        
        params = self._determine_clustering_parameters(embeddings)
        
        self.min_samples = params['min_samples']
        self.max_radius = params['max_radius']
        self.faiss_indexer.similarity_threshold = params['similarity_threshold']
        
        if len(embeddings) < 200:
            results = self._process_small_dataset(embeddings, image_ids)
        else:
            results = self._process_large_dataset(
                embeddings, 
                image_ids, 
                params['n_clusters']
            )
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Failed to process batch: {str(e)}")

  def _process_small_dataset(self, embeddings: np.ndarray, image_ids: List[str]) -> Dict:
    """Process small dataset using hierarchical clustering with subcategories."""
    try:
        from sklearn.cluster import AgglomerativeClustering
        
        n_main_clusters = self._calculate_optimal_clusters(len(embeddings))
        main_clustering = AgglomerativeClustering(
            n_clusters=n_main_clusters,
            metric='euclidean',
            linkage='ward'
        )
        main_assignments = main_clustering.fit_predict(embeddings)
        
        categories = {}
        subcategories = {}
        
        for i in range(n_main_clusters):
            cluster_mask = main_assignments == i
            if np.sum(cluster_mask) >= self.min_samples:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_ids = [image_ids[j] for j, m in enumerate(cluster_mask) if m]
                
                main_category_id = self.faiss_indexer.create_new_category(
                    embedding_vec=np.mean(cluster_embeddings, axis=0).reshape(1, -1),
                    image_id=cluster_ids[0]
                )
                
                self.faiss_indexer.category_metadata['members'][main_category_id] = cluster_ids
                categories[main_category_id] = cluster_ids
                
                self.category_statistics[main_category_id] = {
                    'diversity': self._calculate_diversity(cluster_embeddings),
                    'quality': self._calculate_quality(
                        cluster_embeddings,
                        np.mean(cluster_embeddings, axis=0)
                    )
                }
                
                if len(cluster_ids) >= 3 * self.min_samples:
                    n_subclusters = max(2, min(
                        len(cluster_ids) // self.min_samples,
                        int(np.sqrt(len(cluster_ids)))
                    ))
                    
                    sub_clustering = AgglomerativeClustering(
                        n_clusters=n_subclusters,
                        metric='euclidean',
                        linkage='ward'
                    )
                    sub_assignments = sub_clustering.fit_predict(cluster_embeddings)
                    
                    for j in range(n_subclusters):
                        sub_mask = sub_assignments == j
                        if np.sum(sub_mask) >= self.min_samples:
                            sub_embeddings = cluster_embeddings[sub_mask]
                            sub_ids = [cluster_ids[k] for k, m in enumerate(sub_mask) if m]
                            
                            # Create subcategory
                            subcat_id = self.faiss_indexer.create_new_category(
                                embedding_vec=np.mean(sub_embeddings, axis=0).reshape(1, -1),
                                image_id=sub_ids[0],
                                parent_id=main_category_id
                            )
                            
                            # Update hierarchy
                            self.faiss_indexer.category_metadata['hierarchy'][main_category_id]['children'].append(subcat_id)
                            
                            # Store subcategory members
                            self.faiss_indexer.category_metadata['members'][subcat_id] = sub_ids
                            subcategories[subcat_id] = {
                                'parent': main_category_id,
                                'members': sub_ids
                            }
                            
                            # Update subcategory statistics
                            self.category_statistics[subcat_id] = {
                                'diversity': self._calculate_diversity(sub_embeddings),
                                'quality': self._calculate_quality(
                                    sub_embeddings,
                                    np.mean(sub_embeddings, axis=0)
                                )
                            }
        
        return {
            'categories': categories,
            'subcategories': subcategories,
            'statistics': self.category_statistics
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to process small dataset: {str(e)}")

  def _process_large_dataset(self, embeddings: np.ndarray, image_ids: List[str], 
                         n_clusters: int) -> Dict:
    """Process large dataset using k-means clustering with subcategories."""
    try:
        # First level: Main categories using Faiss k-means
        kmeans = faiss.Kmeans(
            d=embeddings.shape[1],
            k=n_clusters,
            gpu=False,
            niter=50,
            verbose=True
        )
        
        # Normalize embeddings
        normalized_embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
        normalized_embeddings = normalized_embeddings / norms
        
        kmeans.train(normalized_embeddings)
        
        _, main_assignments = kmeans.index.search(normalized_embeddings, 1)
        
        categories = {}
        subcategories = {}
        
        for i in range(n_clusters):
            cluster_mask = main_assignments.flatten() == i
            if np.sum(cluster_mask) >= self.min_samples:
                cluster_embeddings = embeddings[cluster_mask]
                cluster_ids = [image_ids[j] for j, m in enumerate(cluster_mask) if m]
                
                # Create main category
                main_category_id = self.faiss_indexer.create_new_category(
                    embedding_vec=np.mean(cluster_embeddings, axis=0).reshape(1, -1),
                    image_id=cluster_ids[0]
                )
                
                # Store main category members
                self.faiss_indexer.category_metadata['members'][main_category_id] = cluster_ids
                categories[main_category_id] = cluster_ids
                
                # Update main category statistics
                self.category_statistics[main_category_id] = {
                    'diversity': self._calculate_diversity(cluster_embeddings),
                    'quality': self._calculate_quality(
                        cluster_embeddings,
                        np.mean(cluster_embeddings, axis=0)
                    )
                }
                
                if len(cluster_ids) >= 3 * self.min_samples:
                    n_subclusters = max(2, min(
                        len(cluster_ids) // self.min_samples,
                        int(np.sqrt(len(cluster_ids)))
                    ))
                    
                    # Use Faiss k-means for subcategories
                    sub_kmeans = faiss.Kmeans(
                        d=embeddings.shape[1],
                        k=n_subclusters,
                        gpu=False,
                        niter=30,
                        verbose=False
                    )
                    
                    # Normalize subcluster embeddings
                    sub_embeddings = cluster_embeddings.astype(np.float32)
                    sub_norms = np.linalg.norm(sub_embeddings, axis=1, keepdims=True)
                    sub_embeddings = sub_embeddings / sub_norms
                    
                    # Train subcategories
                    sub_kmeans.train(sub_embeddings)
                    
                    # Get subcategory assignments
                    _, sub_assignments = sub_kmeans.index.search(sub_embeddings, 1)
                    
                    # Process each subcluster
                    for j in range(n_subclusters):
                        sub_mask = sub_assignments.flatten() == j
                        if np.sum(sub_mask) >= self.min_samples:
                            sub_cluster_embeddings = cluster_embeddings[sub_mask]
                            sub_ids = [cluster_ids[k] for k, m in enumerate(sub_mask) if m]
                            
                            # Create subcategory
                            subcat_id = self.faiss_indexer.create_new_category(
                                embedding_vec=np.mean(sub_cluster_embeddings, axis=0).reshape(1, -1),
                                image_id=sub_ids[0],
                                parent_id=main_category_id
                            )
                            
                            # Update hierarchy
                            self.faiss_indexer.category_metadata['hierarchy'][main_category_id]['children'].append(subcat_id)
                            
                            # Store subcategory members
                            self.faiss_indexer.category_metadata['members'][subcat_id] = sub_ids
                            subcategories[subcat_id] = {
                                'parent': main_category_id,
                                'members': sub_ids
                            }
                            
                            # Update subcategory statistics
                            self.category_statistics[subcat_id] = {
                                'diversity': self._calculate_diversity(sub_cluster_embeddings),
                                'quality': self._calculate_quality(
                                    sub_cluster_embeddings,
                                    np.mean(sub_cluster_embeddings, axis=0)
                                )
                            }
        
        return {
            'categories': categories,
            'subcategories': subcategories,
            'statistics': self.category_statistics
        }
        
    except Exception as e:
        raise RuntimeError(f"Failed to process large dataset: {str(e)}")
      
  
        
  def process_new_image(self, embedding: np.ndarray, image_id: str) -> str:
    """Process a new image and assign it to an existing category or create a new one."""
    try:
        # Find most similar existing categories
        similarities = []
        for cat_id, centroid in self.faiss_indexer.category_metadata['centroids'].items():
            similarity = float(np.dot(embedding.flatten(), centroid.flatten()))
            similarities.append((cat_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Check if image fits in existing category
        if similarities and similarities[0][1] > self.faiss_indexer.similarity_threshold:
            best_category = similarities[0][0]
            # Add to existing category
            self.faiss_indexer.category_metadata['members'][best_category].append(image_id)
            self._update_category_statistics(best_category)
            return best_category
        else:
            # Create new category
            new_category = self.faiss_indexer.create_new_category(
                embedding_vec=embedding.reshape(1, -1),
                image_id=image_id
            )
            return new_category
            
    except Exception as e:
        raise RuntimeError(f"Failed to process new image: {str(e)}")
      
      
  
  def _initialize_categories(self, embeddings: np.ndarray, image_ids: List[str]):
    """Initialize first categories based on data patterns."""
    try:
        n_clusters = self._calculate_optimal_clusters(len(embeddings))
        print(f"Initializing with {n_clusters} clusters for {len(embeddings)} images")
        
        if len(embeddings) < 200:  
            # Simple centroid-based initialization
            centroid = np.mean(embeddings, axis=0)
            category_id = self.faiss_indexer.create_new_category(
                embedding_vec=centroid.reshape(1, -1),
                image_id=image_ids[0]
            )
            self.category_statistics[category_id] = {
                'diversity': self._calculate_diversity(embeddings),
                'quality': self._calculate_quality(embeddings, centroid)
            }
            return

        # For larger datasets, use k-means
        kmeans = faiss.Kmeans(
            d=embeddings.shape[1],
            k=n_clusters,
            gpu=False,
            niter=50,  # Increased iterations for better convergence
            verbose=False  # Suppress warnings
        )
        kmeans.train(embeddings)
        
        distances, assignments = kmeans.index.search(embeddings, 1)
        
        for i in range(n_clusters):
            self._create_category_from_cluster(
                i, assignments, embeddings, image_ids
            )
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize categories: {str(e)}")
    
  def _calculate_optimal_clusters(self, n_samples: int) -> int:
    """Calculate optimal number of clusters based on dataset size."""
    if n_samples < 200:  # For very small datasets
        return max(2, min(
            n_samples // 20,  # 1 cluster per 20 samples
            8  # Cap at 8 clusters for small datasets
        ))
    elif n_samples < 1000:
        return max(5, min(
            n_samples // 30,
            int(np.sqrt(n_samples))
        ))
    else:
        return max(10, min(
            n_samples // 50,
            int(np.sqrt(n_samples) * 1.5)
        ))

  def _create_category_from_cluster(self, 
                                  cluster_idx: int, 
                                  assignments: np.ndarray,
                                  embeddings: np.ndarray,
                                  image_ids: List[str]):
      """Create a category from a cluster if it meets criteria."""
      mask = assignments.flatten() == cluster_idx
      if np.sum(mask) >= self.min_samples:
          cluster_embeddings = embeddings[mask]
          cluster_ids = [image_ids[j] for j, m in enumerate(mask) if m]
         
          centroid = np.mean(cluster_embeddings, axis=0)
          
          category_id = self.faiss_indexer.create_new_category(
              embedding_vec=centroid.reshape(1, -1),
              image_id=cluster_ids[0]
          )
          
          self.category_statistics[category_id] = {
              'diversity': self._calculate_diversity(cluster_embeddings),
              'quality': self._calculate_quality(cluster_embeddings, centroid)
          }
          
        
  def get_category_info(self, category_id: str) -> Dict:
      """Get detailed information about a category."""
      if category_id not in self.faiss_indexer.category_metadata['centroids']:
          raise ValueError(f"Category {category_id} does not exist")
          
      return {
          'label': self.category_labels.get(category_id, None),
          'members': len(self.faiss_indexer.category_metadata['members'][category_id]),
          'statistics': self.category_statistics.get(category_id, {}),
          'subcategories': self.faiss_indexer.category_metadata['hierarchy'][category_id]['children']
      }
      
  def _optimize_category_structure(self):
      """Optimize current category structure."""
      self._merge_similar_categories()
      self._split_diverse_categories()
      self._cleanup_small_categories()
      self._update_category_statistics()
      
  def _calculate_diversity(self, embeddings: np.ndarray) -> float:
      """Calculate diversity score for a set of embeddings."""
      if len(embeddings) < 2:
          return 0.0
      centroid = np.mean(embeddings, axis=0)
      distances = np.linalg.norm(embeddings - centroid, axis=1)
      return float(np.mean(distances))
    
  def _calculate_quality(self, embeddings: np.ndarray, centroid: np.ndarray) -> float:
      """Calculate quality score for a category."""
      distances = np.linalg.norm(embeddings - centroid, axis=1)
      return float(1.0 / (1.0 + np.mean(distances)))
    
  def _update_category_statistics(self):
      """Update statistics for all categories."""
      for category_id in self.faiss_indexer.category_metadata['centroids']:
          member_embeddings = self._get_member_embeddings(category_id)
          centroid = self.faiss_indexer.category_metadata['centroids'][category_id]
          
          self.category_statistics[category_id] = {
              'diversity': self._calculate_diversity(member_embeddings),
              'quality': self._calculate_quality(member_embeddings, centroid)
          }
          
  def _get_member_embeddings(self, category_id: str) -> np.ndarray:
    member_ids = self.faiss_indexer.category_metadata['members'].get(category_id, [])
    embeddings = []
    for img_id in member_ids:
        if img_id not in self.faiss_indexer.image_ids:
            continue  
        idx = self.faiss_indexer.image_ids.index(img_id)
        embedding = self.faiss_indexer.index.reconstruct(idx)
        
        if embedding is None:
            continue  
        
        embeddings.append(embedding)
    return np.array(embeddings) if embeddings else np.array([])


  def _merge_similar_categories(self):
    """Merge categories that are very similar."""
    try:
        categories = list(self.faiss_indexer.category_metadata['centroids'].keys())
        merged_count = 0
        
        print(f"Starting merge with {len(categories)} categories")
        
        for i, cat1 in enumerate(categories):
            if cat1 not in self.faiss_indexer.category_metadata['centroids']:
                print(f"Skipping category {cat1} - already merged")
                continue
                
            for cat2 in categories[i+1:]:
                if cat2 not in self.faiss_indexer.category_metadata['centroids']:
                    print(f"Skipping category {cat2} - already merged")
                    continue
                
                try:
                    similarity = self.faiss_indexer.category_metadata['relationships']\
                        .get(cat1, {}).get(cat2, 0)
                    
                    print(f"Checking similarity between {cat1} and {cat2}: {similarity}")
                    
                    if similarity > self.faiss_indexer.similarity_threshold:
                        print(f"Merging categories {cat1} and {cat2}")
                        self._merge_categories(cat1, cat2)
                        merged_count += 1
                        print(f"Successfully merged. Current count: {merged_count}")
                        
                except Exception as e:
                    print(f"Warning: Failed to process categories {cat1} and {cat2}: {str(e)}")
                    continue
        
        print(f"Merge process completed. Total merges: {merged_count}")
        return merged_count
        
    except Exception as e:
        print(f"Error in merge_similar_categories: {str(e)}")
        return 0

  def _split_diverse_categories(self):
    """Split categories that are too diverse."""
    split_count = 0  # Initialize counter
    try:
        for category_id in list(self.faiss_indexer.category_metadata['centroids'].keys()):
            members = self.faiss_indexer.category_metadata['members'][category_id]
            if len(members) > 2 * self.min_samples:
                member_embeddings = self._get_member_embeddings(category_id)

                distances = np.linalg.norm(
                    member_embeddings - 
                    self.faiss_indexer.category_metadata['centroids'][category_id],
                    axis=1
                )

                if np.max(distances) > self.max_radius:
                    try:
                        self._split_category(category_id, member_embeddings, members)
                        split_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to split category {category_id}: {str(e)}")
                        continue
                        
        return split_count
        
    except Exception as e:
        print(f"Error in split_diverse_categories: {str(e)}")
        return 0
      
  def _cleanup_small_categories(self):
    """Remove or merge categories that are too small."""
    cleanup_count = 0
    try:
        print("\nStarting category cleanup...")
        print(f"Minimum samples required: {self.min_samples}")
        
        # Get list of categories
        categories = list(self.faiss_indexer.category_metadata['centroids'].keys())
        print(f"Checking {len(categories)} categories")
        
        # Identify small categories
        small_categories = []
        for cat_id in categories:
            members = self.faiss_indexer.category_metadata['members'][cat_id]
            if len(members) < self.min_samples:
                print(f"Category {cat_id} has {len(members)} members (below minimum)")
                small_categories.append(cat_id)
            else:
                print(f"Category {cat_id} has {len(members)} members (OK)")
        
        print(f"Found {len(small_categories)} categories below minimum size")
        
        # Process small categories
        for cat_id in small_categories:
            try:
                # First try to find a similar category to merge with
                best_match = None
                best_similarity = 0
                
                for other_id in categories:
                    if other_id != cat_id and other_id in self.faiss_indexer.category_metadata['centroids']:
                        similarity = self.faiss_indexer.category_metadata['relationships']\
                            .get(cat_id, {}).get(other_id, 0)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = other_id
                
                if best_match and best_similarity > self.faiss_indexer.similarity_threshold * 0.8:
                    print(f"Merging small category {cat_id} into {best_match}")
                    self._merge_categories(cat_id, best_match)
                else:
                    # Only remove if we really can't find a good merge target
                    print(f"No good merge target found for {cat_id}")
                    members = len(self.faiss_indexer.category_metadata['members'][cat_id])
                    if members < self.min_samples:
                        print(f"Removing category {cat_id} with {members} members")
                        self._remove_category(cat_id)
                    else:
                        print(f"Keeping category {cat_id} with {members} members")
                
                cleanup_count += 1
                
            except Exception as e:
                print(f"Error processing small category {cat_id}: {str(e)}")
                continue
        
        remaining_categories = len(self.faiss_indexer.category_metadata['centroids'])
        print(f"\nCleanup complete. {cleanup_count} categories processed")
        print(f"Remaining categories: {remaining_categories}")
        
        return cleanup_count
        
    except Exception as e:
        print(f"Error in cleanup_small_categories: {str(e)}")
        return 0

  def _merge_categories(self, source_id: str, target_id: str):
    """Merge source category into target category."""
    try:
        print(f"Starting merge of {source_id} into {target_id}")
        
        if source_id not in self.faiss_indexer.category_metadata['centroids']:
            raise ValueError(f"Source category {source_id} does not exist")
        if target_id not in self.faiss_indexer.category_metadata['centroids']:
            raise ValueError(f"Target category {target_id} does not exist")
            
        source_members = self.faiss_indexer.category_metadata['members'][source_id]
        target_members = self.faiss_indexer.category_metadata['members'][target_id]
        source_centroid = self.faiss_indexer.category_metadata['centroids'][source_id]
        target_centroid = self.faiss_indexer.category_metadata['centroids'][target_id]

        print(f"Merging {len(source_members)} members into {len(target_members)} members")

        total_members = len(source_members) + len(target_members)
        new_centroid = (
            source_centroid * len(source_members) + 
            target_centroid * len(target_members)
        ) / total_members

        self.faiss_indexer.category_metadata['centroids'][target_id] = new_centroid
        self.faiss_indexer.category_metadata['members'][target_id].extend(source_members)

        if self.faiss_indexer.category_metadata['hierarchy'][source_id]['children']:
            self.faiss_indexer.category_metadata['hierarchy'][target_id]['children'].extend(
                self.faiss_indexer.category_metadata['hierarchy'][source_id]['children']
            )

        for image_id in source_members:
            if image_id in self.faiss_indexer.similarity_scores:
                if source_id in self.faiss_indexer.similarity_scores[image_id]:
                    score = self.faiss_indexer.similarity_scores[image_id].pop(source_id)
                    self.faiss_indexer.similarity_scores[image_id][target_id] = score

        self._remove_category(source_id)
        print(f"Successfully completed merge of {source_id} into {target_id}")

    except Exception as e:
        print(f"Failed to merge categories: {str(e)}")
        raise RuntimeError(f"Failed to merge categories: {str(e)}")

  def _split_category(self, category_id: str, embeddings: np.ndarray, image_ids: List[str]) -> bool:
    """Split a diverse category into subcategories using median split."""
    try:
        print(f"Attempting to split category {category_id}")
        print(f"Input shape: {embeddings.shape}")
        print(f"Number of image IDs: {len(image_ids)}")
        
        if embeddings.shape[0] < 2 * self.min_samples:
            print(f"Not enough samples to split category {category_id}")
            return False
            
        print("Using median split approach")
        
        # Calculate distances from centroid
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        median_distance = np.median(distances)
        
        # Split based on median distance
        close_mask = distances <= median_distance
        far_mask = distances > median_distance
        
        close_size = np.sum(close_mask)
        far_size = np.sum(far_mask)
        
        print(f"Close cluster size: {close_size}")
        print(f"Far cluster size: {far_size}")
        
        # Only proceed if both clusters would be large enough
        if close_size < self.min_samples or far_size < self.min_samples:
            print("Split would create clusters that are too small")
            return False
        
        new_categories = []
        
        if close_size >= self.min_samples:
            close_embeddings = embeddings[close_mask]
            close_image_ids = [image_ids[j] for j, m in enumerate(close_mask) if m]
            close_centroid = np.mean(close_embeddings, axis=0)
            
            close_cat_id = self.faiss_indexer.create_new_category(
                embedding_vec=close_centroid.reshape(1, -1),
                image_id=close_image_ids[0]
            )
            
            # Update members explicitly
            self.faiss_indexer.category_metadata['members'][close_cat_id] = close_image_ids
            
            print(f"Created close category {close_cat_id} with {len(close_image_ids)} members")
            new_categories.append(close_cat_id)
            
            self.category_statistics[close_cat_id] = {
                'diversity': self._calculate_diversity(close_embeddings),
                'quality': self._calculate_quality(close_embeddings, close_centroid)
            }
            
            for img_id in close_image_ids:
                if img_id in self.faiss_indexer.similarity_scores:
                    self.faiss_indexer.similarity_scores[img_id][close_cat_id] = 1.0
        
        # Process far cluster
        if far_size >= self.min_samples:
            far_embeddings = embeddings[far_mask]
            far_image_ids = [image_ids[j] for j, m in enumerate(far_mask) if m]
            far_centroid = np.mean(far_embeddings, axis=0)
            
            far_cat_id = self.faiss_indexer.create_new_category(
                embedding_vec=far_centroid.reshape(1, -1),
                image_id=far_image_ids[0]
            )
            
            self.faiss_indexer.category_metadata['members'][far_cat_id] = far_image_ids
            
            print(f"Created far category {far_cat_id} with {len(far_image_ids)} members")
            new_categories.append(far_cat_id)
            
            self.category_statistics[far_cat_id] = {
                'diversity': self._calculate_diversity(far_embeddings),
                'quality': self._calculate_quality(far_embeddings, far_centroid)
            }
            
            for img_id in far_image_ids:
                if img_id in self.faiss_indexer.similarity_scores:
                    self.faiss_indexer.similarity_scores[img_id][far_cat_id] = 1.0
        
        for cat_id in new_categories:
            member_count = len(self.faiss_indexer.category_metadata['members'][cat_id])
            print(f"Verifying category {cat_id} has {member_count} members")
            if member_count < self.min_samples:
                print(f"Warning: Category {cat_id} has too few members after split")
                return False
        
        if len(new_categories) == 2:
            print(f"Created {len(new_categories)} new categories: {new_categories}")
            print(f"Removing original category: {category_id}")
            self._remove_category(category_id)
            return True
            
        for cat_id in new_categories:
            print(f"Cleaning up partial split: removing {cat_id}")
            self._remove_category(cat_id)
            
        print("Split failed - could not create both categories")
        return False
        
    except Exception as e:
        print(f"Failed to split category: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
        
  def _merge_or_remove_category(self, category_id: str):
      """Merge small category with most similar one or remove if no good match."""
      try:
          best_similarity = 0
          best_match = None
          
          for other_id in self.faiss_indexer.category_metadata['centroids']:
              if other_id != category_id:
                  similarity = self.faiss_indexer.category_metadata['relationships']\
                      .get(category_id, {}).get(other_id, 0)
                  if similarity > best_similarity:
                      best_similarity = similarity
                      best_match = other_id
          if best_match and best_similarity > self.faiss_indexer.similarity_threshold * 0.8:
              self._merge_categories(category_id, best_match)
          else:
              self._remove_category(category_id)
      except Exception as e:
          raise RuntimeError(f"Failed to merge or remove category: {str(e)}")
        
  def _remove_category(self, category_id: str):
    """Remove a category and clean up related metadata."""
    try:
        print(f"Removing category {category_id}")
        print(f"Current categories: {list(self.faiss_indexer.category_metadata['centroids'].keys())}")
        
        if category_id not in self.faiss_indexer.category_metadata['centroids']:
            print(f"Warning: Category {category_id} not found in centroids")
            return
            
        del self.faiss_indexer.category_metadata['centroids'][category_id]
        del self.faiss_indexer.category_metadata['members'][category_id]
        del self.faiss_indexer.category_metadata['hierarchy'][category_id]
        
        # Clean up relationships
        if category_id in self.faiss_indexer.category_metadata['relationships']:
            del self.faiss_indexer.category_metadata['relationships'][category_id]
            
        for other_id in self.faiss_indexer.category_metadata['relationships']:
            if category_id in self.faiss_indexer.category_metadata['relationships'][other_id]:
                del self.faiss_indexer.category_metadata['relationships'][other_id][category_id]
        
        # Clean up statistics
        if category_id in self.category_statistics:
            del self.category_statistics[category_id]
        if category_id in self.category_labels:
            del self.category_labels[category_id]
            
        print(f"Successfully removed category {category_id}")
        print(f"Remaining categories: {list(self.faiss_indexer.category_metadata['centroids'].keys())}")
        
    except Exception as e:
        print(f"Error removing category {category_id}: {str(e)}")
        raise RuntimeError(f"Failed to remove category: {str(e)}")
        
  def get_category_summary(self) -> Dict:
    """Get summary of all categories."""
    categories = self.faiss_indexer.category_metadata['centroids']
    return {
        'total_categories': len(categories),
        'total_images': len(self.faiss_indexer.image_ids),
        'categories': {
            cat_id: {
                'label': self.category_labels.get(cat_id, 'Unlabeled'),
                'size': len(self.faiss_indexer.category_metadata['members'][cat_id]),
                'quality': self.category_statistics[cat_id]['quality'],
                'diversity': self.category_statistics[cat_id]['diversity']
            }
            for cat_id in categories
        }
    }
      
  def find_similar_categories(self, category_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
      """Find categories similar to given category."""
      if category_id not in self.faiss_indexer.category_metadata['centroids']:
          raise ValueError(f"Category {category_id} does not exist")
          
      similarities = []
      for other_id in self.faiss_indexer.category_metadata['centroids']:
          if other_id != category_id:
              sim = self.faiss_indexer.category_metadata['relationships']\
                  .get(category_id, {})\
                  .get(other_id, 0)
              if sim > threshold:
                  similarities.append((other_id, sim))
                  
      return sorted(similarities, key=lambda x: x[1], reverse=True)

  def _validate_category_structure(self):
    """Validate category metadata consistency."""
    try:
        categories = set(self.faiss_indexer.category_metadata['centroids'].keys())
        
        for key, metadata in self.faiss_indexer.category_metadata.items():
            if set(metadata.keys()) != categories:
                print(f"Warning: Inconsistency in {key} metadata")
                print(f"Expected categories: {categories}")
                print(f"Found categories: {set(metadata.keys())}")
        
        for cat_id in categories:
            members = self.faiss_indexer.category_metadata['members'][cat_id]
            if len(members) < self.min_samples:
                print(f"Warning: Category {cat_id} has fewer than minimum samples: {len(members)}")
        
        return True
    except Exception as e:
        print(f"Category validation failed: {str(e)}")
        return False

  def generate_category_names_from_folders(self, category_folders: Dict[str, str]) -> Dict[str, str]:
    """Generate content-driven category names from image folders."""
    try:
        print("\n=== Generating Category Names ===")
        category_names = {}
        
        for category_id, folder_path in category_folders.items():
            try:
                if not isinstance(folder_path, str) or not os.path.exists(folder_path):
                    print(f"\nSkipping invalid folder for category {category_id}")
                    category_names[category_id] = f"Category {category_id}"
                    continue

                image_files = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not image_files:
                    print(f"\nNo valid images found in category {category_id}")
                    category_names[category_id] = f"Category {category_id}"
                    continue

                print(f"\nProcessing category {category_id}")
                print(f"Found {len(image_files)} images")

                descriptions = []
                for img_file in image_files:
                    try:
                        img_path = os.path.join(folder_path, img_file)
                        img = Image.open(img_path).convert('RGB')
                        description = self.vlm_processor.generate_image_description(img)
                        
                        if description and len(description.strip()) > 3:
                            print(f"\nImage: {img_file}")
                            print(f"Description: {description}")
                            descriptions.append(description)
                            
                    except Exception as e:
                        print(f"Error processing {img_file}: {str(e)}")
                        continue

                if descriptions:
                    print("\nAll generated descriptions for this category:")
                    for desc in descriptions:
                        print(f"- {desc}")
                        
                    print("\nGenerating theme from all descriptions...")
                    category_name = self.vlm_processor.synthesize_theme_with_llama(descriptions)
                    
                    if category_name and len(category_name.strip()) > 3:
                        category_name = ' '.join(word.capitalize() for word in category_name.split())
                        category_name = category_name.replace(' And ', ' & ')
                        category_names[category_id] = category_name
                        print(f"\nAssigned name: {category_name}")
                        
                        try:
                            self._rename_category_folder(folder_path, category_name)
                        except Exception as e:
                            print(f"Failed to rename folder: {str(e)}")
                    else:
                        category_names[category_id] = f"Category {category_id}"
                        print(f"\nFalling back to default name: Category {category_id}")
                else:
                    category_names[category_id] = f"Category {category_id}"
                    print(f"\nNo valid descriptions generated, using default: Category {category_id}")

            except Exception as e:
                print(f"\nError processing category {category_id}: {str(e)}")
                category_names[category_id] = f"Category {category_id}"
                continue

            print("\n" + "="*50 + "\n") 

        return category_names

    except Exception as e:
        print(f"Failed to generate category names: {str(e)}")
        raise
      
      
  def _rename_category_folder(self, old_path: str, new_name: str):
    """Rename category folder with proper error handling."""
    try:
        # Create valid folder name
        valid_name = "".join(
            c for c in new_name 
            if c.isalnum() or c in (' ', '-', '_')
        ).strip()
        
        new_path = os.path.join(os.path.dirname(old_path), valid_name)
        
        # Handle naming conflicts
        counter = 1
        original_new_path = new_path
        while os.path.exists(new_path):
            new_path = f"{original_new_path}_{counter}"
            counter += 1
        
        os.rename(old_path, new_path)
        print(f"Renamed folder: {os.path.basename(old_path)} → {os.path.basename(new_path)}")
        
    except Exception as e:
        print(f"Failed to rename folder: {str(e)}")
         
  def organize_images_into_folders(self, source_folder: str, output_base_folder: str) -> Dict[str, str]:
    """Organize images into hierarchical category folders."""
    try:
        print("\nOrganizing images into category folders...")
        
        os.makedirs(output_base_folder, exist_ok=True)
        moved_files = 0
        errors = 0
        category_folders = {}  # Track category ID to folder path mapping
        
        # Process main categories first
        for category_id, members in self.faiss_indexer.category_metadata['members'].items():
            try:
                parent_id = self.faiss_indexer.category_metadata['hierarchy'][category_id]['parent']
                
                if parent_id is None:  # Main category
                    # Create main category folder
                    folder_name = f"category_{category_id}"
                    category_folder = os.path.join(output_base_folder, folder_name)
                    os.makedirs(category_folder, exist_ok=True)
                    category_folders[category_id] = category_folder
                    
                    # Handle subcategories
                    subcategories = self.faiss_indexer.category_metadata['hierarchy'][category_id]['children']
                    if subcategories:
                        for subcat_id in subcategories:
                            subcat_folder = os.path.join(category_folder, f"subcat_{subcat_id}")
                            os.makedirs(subcat_folder, exist_ok=True)
                            category_folders[subcat_id] = subcat_folder
                            
                            # Copy subcategory images
                            subcat_members = self.faiss_indexer.category_metadata['members'][subcat_id]
                            for tensor_id in subcat_members:
                                try:
                                    if self._copy_image(tensor_id, source_folder, subcat_folder):
                                        moved_files += 1
                                except Exception as e:
                                    errors += 1
                                    continue
                    else:
                        # Copy main category images
                        for tensor_id in members:
                            try:
                                if self._copy_image(tensor_id, source_folder, category_folder):
                                    moved_files += 1
                            except Exception as e:
                                errors += 1
                                continue
                    
            except Exception as e:
                print(f"Error processing category {category_id}: {str(e)}")
                continue
        
        print(f"\nImage organization complete:")
        print(f"Successfully organized {moved_files} images")
        print(f"Encountered {errors} errors")
        
        return category_folders
        
    except Exception as e:
        print(f"Failed to organize images: {str(e)}")
        raise

  def _copy_image(self, tensor_id: str, source_folder: str, dest_folder: str):
    """Helper method to copy an image to destination folder."""
    image_name = tensor_id.replace('.pt', '')
    for ext in ['.jpg', '.jpeg', '.png']:
        source_path = os.path.join(source_folder, image_name + ext)
        if os.path.exists(source_path):
            dest_path = os.path.join(dest_folder, image_name + ext)
            shutil.copy2(source_path, dest_path)
            return True
    return False
