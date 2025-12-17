import gc
import torch


class ModelRegistry:
    """Registry to manage multiple loaded models.
    
    This allows loading multiple models in a single session and
    switching between them without reloading, while also providing
    memory management capabilities.
    """
    
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_key: str, model_name: str, token: str, use_4bit: bool = True):
        from .wrappers import ModelHelper
        
        if model_key in self.models:
            print(f"Model '{model_key}' already loaded.")
            return
        
        print(f"\n{'='*60}")
        print(f"Loading model: {model_key}")
        print(f"{'='*60}")
        
        self.models[model_key] = ModelHelper(model_name, token, use_4bit)
        
        print(f"{'='*60}")
        print(f"{model_key} ready")
        print(f"{'='*60}\n")
    
    def get_model(self, model_key: str):
        if model_key not in self.models:
            available = list(self.models.keys())
            raise ValueError(
                f"Model '{model_key}' not loaded in registry.\n"
                f"Available models: {available if available else 'None'}\n"
                f"Use registry.load_model() to load a model first."
            )
        return self.models[model_key]
    
    def list_models(self):
        return list(self.models.keys())
    
    def unload_model(self, model_key: str):
        if model_key in self.models:
            del self.models[model_key]
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Model '{model_key}' unloaded and memory freed")
        else:
            print(f"Model '{model_key}' not found in registry")
    
    def unload_all(self):
        model_keys = list(self.models.keys())
        for key in model_keys:
            self.unload_model(key)
        print(f"All models unloaded")
    
    def get_memory_info(self):
        memory_info = {}
        for key, model_helper in self.models.items():
            memory_info[key] = {
                'name': model_helper.model_name,
                'memory_gb': model_helper.model.get_memory_footprint() / 1e9,
                'num_layers': model_helper.num_layers,
                'vocab_size': model_helper.vocab_size,
                'quantized': model_helper.use_4bit
            }
        return memory_info