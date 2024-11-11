from models.clip_model import CLIP_MODEL
from models.faiss_model import FAISSIndexer
from clustering.categorymanager import CategoryManager
import sys
import os
from typing import Dict, Any, List
from preprocessing.preprocessing import preprocess_data
from embeddings.embeddings import generate_embeddings_using_model

def create_directories(config: Dict[str, Any]) -> None:
    """Create necessary directories if they don't exist."""
    directories = [
        config['NORMALIZED_DATA_PATH'],
        os.path.dirname(config['FAISS_INDEX_PATH']),
        os.path.dirname(config['IMAGE_IDS_PATH']),
        config['CLUSTER_OUTPUT_PATH']
    ]
    
    for directory in directories:
        if not directory:
            continue
        os.makedirs(directory, exist_ok=True)
        if os.path.isdir(directory):
            print(f"Directory created successfully: {directory}")
        else:
            print(f"Failed to create directory: {directory}")


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters."""
    required_fields = [
        'IMAGE_FOLDER',
        'NORMALIZED_DATA_PATH',
        'FAISS_INDEX_PATH',
        'IMAGE_IDS_PATH'
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")
    
    # Validate image folder
    if not os.path.exists(config['IMAGE_FOLDER']):
        raise FileNotFoundError(f"Image folder not found: {config['IMAGE_FOLDER']}")
    
    # Check image folder contains images
    image_files = [f for f in os.listdir(config['IMAGE_FOLDER']) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError(f"No image files found in {config['IMAGE_FOLDER']}")
    
    # Validate thresholds
    if config.get('SIMILARITY_THRESHOLD', 0.75) <= 0 or config.get('SIMILARITY_THRESHOLD', 0.75) >= 1:
        raise ValueError("SIMILARITY_THRESHOLD must be between 0 and 1")
    
    if config.get('MIN_SAMPLES', 2) < 2:
        raise ValueError("MIN_SAMPLES must be at least 2")

def count_images(folder_path: str) -> int:
    """Count number of images in folder."""
    return len([f for f in os.listdir(folder_path) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

def run_full_workflow(config: Dict[str, Any]) -> None:
    """Run workflow with comprehensive error handling and progress tracking."""
    try:
        print("\n=== Initializing Workflow ===")
        validate_config(config)
        create_directories(config)
        
        # Count images
        num_images = count_images(config['IMAGE_FOLDER'])
        print(f"Found {num_images} images in dataset")
        
        if num_images > 200:
            print("Warning: Current configuration is optimized for ≤200 images")
        
        # Initialize models
        print("\n=== Initializing Models ===")
        try:
            clip_model = CLIP_MODEL()
            print("CLIP model initialized successfully")
            
            faiss_indexer = FAISSIndexer(
                dim=512,
                similarity_threshold=config.get('SIMILARITY_THRESHOLD', 0.7),
                max_categories_per_image=config.get('MAX_CATEGORIES', 3)
            )
            print("FAISS indexer initialized successfully")
            
            category_manager = CategoryManager(
                faiss_indexer=faiss_indexer,
                min_samples_category=config.get('MIN_SAMPLES', 2),
                max_category_radius=config.get('MAX_RADIUS', 0.3)
            )
            print("Category manager initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
        
        # Step 1: Preprocess images
        print("\n=== Step 1: Preprocessing Images ===")
        try:
            processed, errors = preprocess_data(
                config['IMAGE_FOLDER'],
                config['NORMALIZED_DATA_PATH']
            )
            
            if processed == 0:
                raise RuntimeError("No images were successfully processed")
            
            print(f"Successfully preprocessed {processed} images ({errors} errors)")
            
        except Exception as e:
            raise RuntimeError(f"Preprocessing failed: {str(e)}")
        
        # Step 2: Generate embeddings
        print("\n=== Step 2: Generating Embeddings ===")
        try:
            embeddings, image_ids = generate_embeddings_using_model(
                config['NORMALIZED_DATA_PATH']
            )
            
            if embeddings is None or len(embeddings) == 0:
                raise RuntimeError("No embeddings were generated")
            
            print(f"Generated embeddings for {len(image_ids)} images")
            print(f"Embedding shape: {embeddings.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
        
        # Step 3: Dynamic Categorization
        print("\n=== Step 3: Processing Categories ===")
        try:
            results = category_manager.process_batch(
                embeddings, 
                image_ids,
                config['IMAGE_FOLDER']  
            )

            if not results:
                raise RuntimeError("No categorization results were generated")

        except Exception as e:
            raise RuntimeError(f"Categorization failed: {str(e)}")
        
        # Print category summary
        try:
            summary = category_manager.get_category_summary()
            print("\nCategory Summary:")
            print(f"Total Categories: {summary['total_categories']}")
            print(f"Total Images: {summary['total_images']}")
            
            for cat_id, info in summary['categories'].items():
                print(f"\nCategory {cat_id}:")
                print(f"  Label: {info['label'] if info.get('label') else 'Unlabeled'}")
                print(f"  Size: {info['size']} images")
                print(f"  Quality Score: {info['quality']:.3f}")
                print(f"  Diversity Score: {info['diversity']:.3f}")
            
        except Exception as e:
            print(f"Warning: Failed to generate summary: {str(e)}")
        
        print("\n=== Organizing Images into Categories ===")
        try:
            # First organize into folders
            category_folders = category_manager.organize_images_into_folders(
                source_folder=config['IMAGE_FOLDER'],
                output_base_folder=config['CLUSTER_OUTPUT_PATH']
            )
            
            if category_folders:  # Check if we have valid folders
                print("\n=== Generating Category Names ===")
                category_names = category_manager.generate_category_names_from_folders(category_folders)
                
                if category_names:  # Check if we got valid names
                    # Update category labels
                    category_manager.category_labels.update(category_names)
                    print(f"Generated names for {len(category_names)} categories")
            else:
                print("Warning: No category folders were created")
                
        except Exception as e:
            print(f"Warning: Failed to organize and name categories: {str(e)}")
            print(f"Error details: {type(e).__name__}")
            import traceback
            traceback.print_exc()
        
        # Save results
        print("\n=== Saving Results ===")
        try:
            faiss_indexer.save_index(config['FAISS_INDEX_PATH'])
            print(f"Saved index to: {config['FAISS_INDEX_PATH']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to save results: {str(e)}")
        
        print("\n=== Workflow Complete ===")
        print(f"Processed {processed} images with {errors} errors")
        print(f"Created {summary['total_categories']} categories")
        
    except Exception as e:
        print(f"\nCritical workflow error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Configuration for small dataset
        config = {
              'IMAGE_FOLDER': "/Users/tilakpatel/Desktop/later-data-practicum/images",
              'NORMALIZED_DATA_PATH': "/Users/tilakpatel/Desktop/later-data-practicum/normalized_data",
              'FAISS_INDEX_PATH': "faiss_index/index_file.index",
              'IMAGE_IDS_PATH': "faiss_index/image_ids.pkl",
              'CLUSTER_OUTPUT_PATH': "clustered_images"
        }
        
        # Create project root if it doesn't exist
        project_root = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(project_root, exist_ok=True)
        
        print(f"Starting workflow from: {project_root}")
        run_full_workflow(config)
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)
