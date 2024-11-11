import clip
import torch
from typing import Tuple, Any

class CLIP_MODEL:
    def __init__(self, version="ViT-B/32"):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load(version, device=self.device)
            print(f"CLIP model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CLIP model: {str(e)}")
    
    def load_clip_model(self) -> Tuple[Any, Any, str]:
        """Return the model, preprocess function, and device."""
        try:
            return self.model.to(self.device), self.preprocess, self.device
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model components: {str(e)}")

    def encode_image(self, image):
        """Encode image and normalize embedding."""
        if image is None:
            raise ValueError("Image input cannot be None")
        try:
            with torch.no_grad():
                embedding = self.model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                return embedding
        except Exception as e:
            raise RuntimeError(f"Failed to encode image: {str(e)}")
