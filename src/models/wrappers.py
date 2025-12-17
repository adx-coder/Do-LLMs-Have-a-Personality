import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class BlockOutputWrapper(nn.Module):
    """Wrapper that captures unembedded logits and hidden states.
    
    This wrapper intercepts the output of each transformer layer and:
    1. Stores the hidden states for later analysis
    2. Projects hidden states through the language model head to get logits
    3. Passes through the original layer output unchanged
    """
    
    def __init__(self, block, lm_head, norm):
        super().__init__()
        self.block = block
        self.lm_head = lm_head
        self.norm = norm
        self._reset_caches()
    
    def _reset_caches(self):
        self.hidden_state = None
        self.block_output_unembedded = None
    
    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs) 
        hidden_states = output[0] if isinstance(output, tuple) else output 
        self.hidden_state = hidden_states.detach().cpu()  
        self.block_output_unembedded = self.lm_head(self.norm(hidden_states))  
        return output
    
    def reset(self):
        self._reset_caches()
    
    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)


class ModelHelper:
    def __init__(self, model_name: str, token: str = None, use_4bit: bool = True):
        print(f"Loading model: {model_name}")
        print(f"4-bit quantization: {use_4bit}")
        
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=token,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        self._wrap_layers()
        
        self.num_layers = len(self.model.model.layers)
        self.vocab_size = self.model.config.vocab_size
        
        print(f"Model loaded: {self.num_layers} layers, {self.vocab_size:,} vocab")
        print(f"Memory: {self.model.get_memory_footprint() / 1e9:.2f} GB\n")
    
    def _wrap_layers(self):
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm
            )