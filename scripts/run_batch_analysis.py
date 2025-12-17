"""Script to run analysis on multiple models."""
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.registry import ModelRegistry
from utils.data_loader import load_prompt_pairs
from analysis.causal_tracer import run_batch_causal_tracing
from analysis.logit_attribution import run_attribution_analysis
from analysis.tuned_lens import run_tuned_lens_analysis
from visualization.causal_plots import visualize_causal_tracing
from visualization.attribution_plots import visualize_attribution
from visualization.lens_plots import visualize_tuned_lens


def main():
    with open('config/models.json', 'r') as f:
        MODELS = json.load(f)
        MODEL_KEYS = ['afm', 'llama', 'mixtral']  ## only these models for now
    
    registry = ModelRegistry()
    
    prompt_pairs = load_prompt_pairs()
    
    for model_key in MODEL_KEYS:
        print(f"\n ANALYZING MODEL: {model_key} \n")
        
        registry.load_model(
            model_key,
            MODELS[model_key]['name'],
            MODELS[model_key]['token'],
            use_4bit=True
        )
        
        print(f"\nCausal Tracing")
        df_layers, df_summary = run_batch_causal_tracing(
            model_key, prompt_pairs, registry
        )
        visualize_causal_tracing(df_layers, df_summary, model_key)
        
        print(f"\nDirect Logit Attribution")
        df_attr = run_attribution_analysis(
            model_key, prompt_pairs, registry
        )
        visualize_attribution(df_attr, model_key)
        
        print(f"\nTuned Lens Analysis")
        predictions, trajectories = run_tuned_lens_analysis(
            model_key, prompt_pairs, registry
        )
        visualize_tuned_lens(trajectories, model_key)
        
        print(f"\nUnloading {model_key}")
        registry.unload_model(model_key)
        
        print(f"\n COMPLETED: {model_key}\n")
    
    print("\nAll models analyzed successfully!")


if __name__ == "__main__":
    main()