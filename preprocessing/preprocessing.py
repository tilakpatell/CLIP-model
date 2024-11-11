import sys
import os
import torch
from PIL import Image
from typing import Tuple
from models.clip_model import CLIP_MODEL

sys.path.append("/Users/tilakpatel/Desktop/later-data-practicum")

def validate_image(image_path: str) -> bool:
    """Validate if image file is readable and valid."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

def preprocess_data(folder_path: str, output_path: str) -> Tuple[int, int]:
    """Preprocesses images with comprehensive error handling and validation."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Image folder not found: {folder_path}")
    
    try:
        # Initialize CLIP model
        clip_model_instance = CLIP_MODEL()
        model, preprocess, device = clip_model_instance.load_clip_model()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize CLIP model: {str(e)}")
    
    os.makedirs(output_path, exist_ok=True)
    
    supported_formats = {'.jpg', '.jpeg', '.png'}
    processed_count = 0
    error_count = 0
    
    # Get total number of valid images for progress tracking
    total_images = sum(1 for f in os.listdir(folder_path) 
                      if any(f.lower().endswith(fmt) for fmt in supported_formats))
    
    print(f"Found {total_images} images to process")
    
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(fmt) for fmt in supported_formats):
            full_image_path = os.path.join(folder_path, filename)
            
            try:
                # Validate image before processing
                if not validate_image(full_image_path):
                    raise ValueError("Invalid or corrupted image file")
                
                # Load and preprocess image
                with Image.open(full_image_path) as pil_image:
                    pil_image = pil_image.convert("RGB")
                    preprocessed_tensor = preprocess(pil_image).unsqueeze(0).to(device)
                    
                    # Save preprocessed tensor
                    output_final_path = os.path.join(
                        output_path, 
                        f"{os.path.splitext(filename)[0]}.pt"
                    )
                    
                    torch.save(preprocessed_tensor, output_final_path)
                    processed_count += 1
                    
                    # Print progress every 10%
                    if processed_count % max(1, total_images // 10) == 0:
                        print(f"Processed {processed_count}/{total_images} images")
                        
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_count} images")
    print(f"Errors encountered: {error_count} images")
    
    return processed_count, error_count

if __name__ == "__main__":
    try:
        folder_path = sys.argv[1] if len(sys.argv) > 1 else '/Users/tilakpatel/Desktop/later-data-practicum/images'
        output_path = sys.argv[2] if len(sys.argv) > 2 else '/Users/tilakpatel/Desktop/later-data-practicum/normalized_data'
        
        processed, errors = preprocess_data(folder_path, output_path)
        
        if errors > processed:
            sys.exit(1)  # Exit with error if more failures than successes
            
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)
