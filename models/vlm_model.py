import logging
from typing import List
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import os
from llama_cpp import Llama


class VLMProcessor:
    """Vision Language Model Processor for dynamic theme extraction."""
    
    def __init__(self):
      """Initialize models and processors."""
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      self._setup_logging()
      self._initialize_models()
      self._initialize_llama() 
      self.metrics = {
          'successful_descriptions': 0,
          'failed_descriptions': 0,
          'total_processing_time': 0,
          'average_processing_time': 0
      }

    def _initialize_llama(self):
      """Initialize Mistral model."""
      try:
          model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "mistral-7b-instruct-v0.2.Q5_K_M.gguf" 
          )

          self.llama = Llama(
              model_path=model_path,
              n_ctx=2048,         
              n_threads=4,         
              n_gpu_layers=1,      
              n_batch=64,          
              verbose=False
          )
          self.logger.info("Mistral model initialized successfully")

      except Exception as e:
          self.logger.error(f"Failed to initialize Mistral: {str(e)}")
          self.llama = None

    def _setup_logging(self):
        """Configure logging."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def _initialize_models(self):
        """Initialize ML models."""
        try:
            self.logger.info(f"Using device: {self.device}")
            
            # Initialize BLIP model
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            ).to(self.device)
            
            # Set to evaluation mode
            self.blip_model.eval()
            
            self.logger.info("Models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {str(e)}")
            raise RuntimeError("Critical model initialization failure")

      
    def generate_image_description(self, image: Image.Image) -> str:
      try:
          if not isinstance(image, Image.Image):
              raise ValueError("Input must be a PIL Image")

          inputs = self.blip_processor(
              images=image, 
              return_tensors="pt",
              padding='max_length',
              max_length=500,
              truncation=True
          ).to(self.device)

          with torch.no_grad():
              outputs = self.blip_model.generate(
                  **inputs,
                  max_new_tokens=500,
                  min_length=5,
                  num_beams=5,
                  temperature=0.7,
                  length_penalty=1.0,
                  early_stopping=True
              )

              description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
              return description

      except Exception as e:
          print(f"Error generating description: {str(e)}")
          return ""
        

    def cleanup(self):
        """Cleanup resources and GPU memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    def synthesize_theme_with_llama(self, descriptions: List[str]) -> str:
      try:
          if not descriptions:
              print("No descriptions provided to Mistral")
              return ""

          print("\nDescriptions for categorization:")
          for desc in descriptions:
              print(f"- {desc}")

          prompt = """<s>[INST] You are an expert at categorizing images. Based on these image descriptions, suggest a clear category name.

          The descriptions represent images from the same group. Create a 2-3 word category name that best describes their common theme.

          Image descriptions:
          {}

          Rules:
          - Use 2-3 clear, descriptive words
          - Capture the main theme or content type
          - Be specific but not too narrow
          - Use "&" instead of "and"
          - Avoid generic words like "images" or "collection"

          Previous successful examples:
          Lifestyle & Portraits
          Inspirational Quotes
          Urban Architecture
          Fashion & Style
          Mixed Items
          Women
          Men
          Dogs
          Cats
          Ads
          Business & Marketing

          Your category name: [/INST]""".format("\n".join(f"• {d}" for d in descriptions))

          response = self.llama(
              prompt,
              max_tokens=10,
              temperature=0.3,      
              top_p=0.9,
              top_k=40,
              repeat_penalty=1.1,
              stop=[".", "\n", ":"],
              echo=False
          )

          if response['choices']:
              return response['choices'][0]['text']
          return ""

      except Exception as e:
          print(f"Theme generation failed: {str(e)}")
          return ""
