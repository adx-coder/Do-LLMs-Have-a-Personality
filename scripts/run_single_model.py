import sys
import json
from pathlib import Path

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
    
    config_path = Path(__file__).parent.parent / 'config' / 'models.json'
    with open(config_path, 'r') as f:
        MODELS = json.load(f)
    
    MODEL_KEY = 'x'  # we changed this at the last minute before pushing it to github
    USE_4BIT = False     # we recommed False for single model runs and True for batch runs and large models
    NUM_PROMPTS = None   # Set to limit number of prompts (None = all)
    
    print(f"SINGLE MODEL ANALYSIS")
    print(f"Model: {MODEL_KEY}")
    print(f"4-bit Quantization: {USE_4BIT}")
    
    registry = ModelRegistry()
    
    try:
        registry.load_model(
            MODEL_KEY,
            MODELS[MODEL_KEY]['name'],
            MODELS[MODEL_KEY]['token'],
            use_4bit=USE_4BIT
        )
    except KeyError:
        print(f"Error: Model '{MODEL_KEY}' not found in config/models.json")
        print(f"Available models: {list(MODELS.keys())}")
        return
    
    prompt_pairs = load_prompt_pairs()
    
    if NUM_PROMPTS is not None:
        prompt_pairs = prompt_pairs[:NUM_PROMPTS]
        print(f"Using first {NUM_PROMPTS} prompts")
    
    try:
        print("\n RUNNING CAUSAL TRACING \n")
        
        df_layers, df_summary = run_batch_causal_tracing(
            MODEL_KEY, prompt_pairs, registry
        )
        
        print("\nGenerating visualizations...")
        visualize_causal_tracing(df_layers, df_summary, MODEL_KEY)
        
        print("\nRUNNING DIRECT LOGIT ATTRIBUTION\n")
        
        df_attr = run_attribution_analysis(
            MODEL_KEY, prompt_pairs, registry
        )
        
        visualize_attribution(df_attr, MODEL_KEY)
        
        print(f"\n{'='*80}")
        print("RUNNING TUNED LENS ANALYSIS")
        print(f"{'='*80}\n")
        
        predictions, trajectories = run_tuned_lens_analysis(
            MODEL_KEY, prompt_pairs, registry
        )
        
        visualize_tuned_lens(trajectories, MODEL_KEY)
        
        # Summary
        print("\n ANALYSIS COMPLETE \n")
        print(f"\n All analyses completed successfully for {MODEL_KEY}")
        print(f"Results saved in respective directories:")
        print(f"- results/causal_tracing/")
        print(f"- results/attribution/")
        print(f"- results/tuned_lens/")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        print("\nCleaning up...")
        registry.unload_all()
        print("Done")


if __name__ == "__main__":
    main()