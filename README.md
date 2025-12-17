# Do LLMs Have a Personality?

A comprehensive toolkit for analyzing how language models process ethical dilemmas through three powerful interpretability techniques. This project helps researchers understand *where* and *how* models make ethical decisions.

---

## Overview

This toolkit implements three state-of-the-art mechanistic interpretability techniques:

1. **Causal Tracing**: Identifies which layers causally influence ethical decisions
2. **Direct Logit Attribution (DLA)**: Decomposes model outputs into attention and MLP contributions  
3. **Tuned Lens**: Tracks how predictions evolve across transformer layers

### Why This Matters

Understanding how AI models make ethical decisions is crucial for:
- **Transparency**: See inside the "black box"
- **Safety**: Identify potential failure modes
- **Research**: Advance our understanding of neural decision-making
- **Alignment**: Ensure models reason about ethics as intended

---

## Project Structure

```
mechanistic-interpretability/
├── config/
│   ├── models.json.template    # Template for model configurations
│   └── models.json             # Your actual tokens (git-ignored)
│
├── data/
│   ├── base_prompts.json       # Original ethical dilemmas
│   └── counterfactual_prompts.json  # Modified scenarios
│
├── results/                    # All analysis outputs go here
│   ├── causal_tracing/        # Causal scores and visualizations
│   ├── attribution/           # Component contribution analysis
│   └── tuned_lens/            # Layer-wise prediction evolution
│
├── src/                       # Core package code
│   ├── models/               # Model loading and management
│   │   ├── wrappers.py       # Intercepts layer outputs
│   │   └── registry.py       # Manages multiple models
│   │
│   ├── analysis/             # Analysis implementations
│   │   ├── causal_tracer.py  # Activation patching
│   │   ├── logit_attribution.py  # Attention/MLP decomposition
│   │   └── tuned_lens.py     # Layer prediction tracking
│   │
│   ├── utils/                # Helper functions
│   │   ├── data_loader.py    # Load prompt pairs
│   │   ├── tokenizer_utils.py # Token ID extraction
│   │   └── metrics.py        # Statistical calculations
│   │
│   └── visualization/        # Plotting functions
│       ├── causal_plots.py
│       ├── attribution_plots.py
│       └── lens_plots.py
│
├── scripts/                  # Executable scripts
│   ├── run_single_model.py   # Analyze one model
│   └── run_batch_analysis.py # Analyze multiple models
│
├── notebooks/                # Interactive tutorials
│   ├── 01_introduction.ipynb
│   ├── 02_theory.ipynb
│   └── ... (more tutorials)
│
├── requirements.txt          # Python dependencies
├── setup.py                 # Package installation
└── README.md               # You are here!
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/mechanistic-interpretability.git
cd mechanistic-interpretability
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n mech-interp python=3.9
conda activate mech-interp
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# OR install as editable package
pip install -e .
```

### Step 4: Configure Your Models

```bash
# Copy the template
cp config/models.json.template config/models.json

# Edit with your HuggingFace tokens
nano config/models.json  # or use any text editor
```

Your `config/models.json` should look like:

```json
{
  "gpt2": {
    "name": "openai-community/gpt2",
    "token": "hf_YourActualTokenHere"
  },
  "llama": {
    "name": "meta-llama/Llama-3.1-8B",
    "token": "hf_YourActualTokenHere"
  }
}
```

**Get your token:** https://huggingface.co/settings/tokens

### Step 5: Prepare Your Data

Place your prompt files in the `data/` directory:

**`data/base_prompts.json`:**
```json
[
  {
    "prompt_id": "scenario_001",
    "prompt": "A doctor must choose between saving 5 patients or 1 patient. Option 1: Save 5. Option 2: Save 1.",
    "original_id": "trolley_001",
    "ethical_option": "1"
  }
]
```

**`data/counterfactual_prompts.json`:**
```json
[
  {
    "prompt_id": "scenario_001",
    "prompt": "A doctor must choose between saving 1 patient or 5 patients. Option 1: Save 1. Option 2: Save 5.",
    "ethical_option": "2"
  }
]
```

---

## Quick Start

### Option 1: Run Single Model Analysis (Recommended for First Time)

```bash
python scripts/run_single_model.py
```

This will:
1. Load a single chosen model
2. Run all three analyses
3. Generate visualizations
4. Save results to `results/` directories

**Expected runtime:** 5-6 hours on a ~46B parameter model (Mixtral-8x7b) or 40-50 mins on Llama 3.1 8B 


## Understanding the Analysis Methods

### 1. Causal Tracing (Activation Patching)

**What it does:** Identifies which layers are causally responsible for the model's ethical decision.

**How it works:**
1. Run model on original prompt → Get choice "A"
2. Run model on counterfactual prompt → Get choice "B"  
3. For each layer:
   - Take activations from counterfactual
   - Patch them into original forward pass
   - Measure how much the output changes

**Key insight:** Large changes indicate that layer is causally important for the decision.

```python
# Pseudocode
base_output = model(base_prompt)  # Choice: 1
cf_hidden = model.get_hidden_states(counterfactual_prompt)

for layer in range(num_layers):
    patched_output = model(base_prompt, patch_layer=layer, patch_state=cf_hidden[layer])
    causal_score[layer] = base_output - patched_output
```

**Output:** CSV with layer-wise causal scores + visualization showing peak layers

### 2. Direct Logit Attribution (DLA)

**What it does:** Decomposes the final prediction into contributions from:
- Attention heads (communication between tokens)
- MLP layers (individual token processing)

**How it works:**
1. Forward through model up to target layer
2. Isolate attention output: `attn_out = layer.attention(hidden_states)`
3. Isolate MLP output: `mlp_out = layer.mlp(hidden_states)`  
4. Project each to vocabulary: `contribution = hidden_state · W_vocab[token_id]`

**Key insight:** Shows whether attention or MLPs drive the ethical decision.

```python
# Pseudocode
hidden = forward_to_layer(prompt, layer=20)

attn_contribution = dot(layer.attention(hidden), vocab_weight[token_1])
mlp_contribution = dot(layer.mlp(hidden), vocab_weight[token_1])
```

**Output:** CSV with component contributions + heatmaps + ratio plots

### 3. Tuned Lens

**What it does:** Shows what the model "thinks" at each layer by projecting intermediate hidden states to vocabulary.

**How it works:**
1. Forward through layers one at a time
2. At each layer, apply final layer norm
3. Project through language model head
4. Get top-k predicted tokens

**Key insight:** Reveals when the model "decides" on its ethical choice (early vs late layers).

```python
# Pseudocode
hidden = embed(prompt)

for layer in model.layers:
    hidden = layer(hidden)
    predictions = lm_head(layer_norm(hidden))
    top_tokens[layer] = predictions.topk(k=10)
```

**Output:** JSON with predictions per layer + trajectory plots showing logit evolution

---

## How to Run the Code

### Basic Workflow

```python
from src.models.registry import ModelRegistry
from src.utils.data_loader import load_prompt_pairs
from src.analysis.causal_tracer import run_batch_causal_tracing
from src.visualization.causal_plots import visualize_causal_tracing

# 1. Initialize registry
registry = ModelRegistry()

# 2. Load a model
registry.load_model(
    'gpt2',                           # Your identifier
    'openai-community/gpt2',          # HuggingFace model name
    'hf_token',                       # Your token
    use_4bit=False                    # GPT-2 is small, no quantization needed
)

# 3. Load data
prompt_pairs = load_prompt_pairs(
    base_path='data/base_prompts.json',
    cf_path='data/counterfactual_prompts.json'
)

# 4. Run analysis
df_layers, df_summary = run_batch_causal_tracing(
    'x',           # Model key
    prompt_pairs,     # Data
    registry          # Registry
)

# 5. Visualize
visualize_causal_tracing(df_layers, df_summary, 'gpt2')
```

### Customizing Analysis

#### Analyze Specific Layers (DLA)

```python
from src.analysis.logit_attribution import run_attribution_analysis

# Analyze only middle and late layers
target_layers = [15, 20, 25, 30, 31]

df_attr = run_attribution_analysis(
    'llama',
    prompt_pairs,
    registry,
    target_layers=target_layers
)
```

#### Limit Number of Prompts

```python
# Analyze only first 50 prompts
prompt_pairs = load_prompt_pairs()
df_layers, df_summary = run_batch_causal_tracing(
    'mixtral8x7b',
    prompt_pairs[:50],  # Slice the list
    registry
)
```

#### Change Output Directory

```python
df_layers, df_summary = run_batch_causal_tracing(
    'mixtral8x7b',
    prompt_pairs,
    registry,
    output_dir='my_results/causal'  # Custom directory
)
```

---

## Project Flow

### High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    1. INITIALIZATION                         │
│  - Load model configurations from config/models.json        │
│  - Initialize ModelRegistry                                  │
│  - Load prompts from data/                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                    2. MODEL LOADING                          │
│  - ModelRegistry loads model via ModelHelper                │
│  - Apply 4-bit quantization (if enabled)                    │
│  - Wrap layers with BlockOutputWrapper                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                  3. CAUSAL TRACING                           │
│  For each prompt pair:                                       │
│    - Get baseline logits on original prompt                 │
│    - Get hidden states from counterfactual                  │
│    - For each layer:                                         │
│        • Patch counterfactual activations into original     │
│        • Measure change in output (causal score)            │
│    - Save layer data and summary statistics                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│              4. DIRECT LOGIT ATTRIBUTION                     │
│  For selected layers:                                        │
│    - Forward to target layer                                │
│    - Isolate attention contributions                        │
│    - Isolate MLP contributions                              │
│    - Project to vocabulary space                            │
│    - Calculate component-wise logits                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                  5. TUNED LENS ANALYSIS                      │
│  For each prompt:                                            │
│    - Forward layer-by-layer                                 │
│    - At each layer, project to vocabulary                   │
│    - Track top-k predictions                                │
│    - Record logit trajectories for target tokens            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                    6. VISUALIZATION                          │
│  Generate plots:                                             │
│    - Causal score evolution across layers                   │
│    - Component contribution heatmaps                         │
│    - Token prediction trajectories                          │
│    - Statistical summaries                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│                      7. OUTPUT                               │
│  Save results:                                               │
│    - CSV files: Detailed numerical results                  │
│    - PNG files: Visualizations                              │
│    - JSON files: Tuned lens predictions                     │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Code Flow

#### When You Run `python scripts/run_single_model.py`

1. **Configuration Loading** (`main()` function)
   ```python
   config_path = Path(__file__).parent.parent / 'config' / 'models.json'
   with open(config_path, 'r') as f:
       MODELS = json.load(f)
   ```

2. **Model Registration** (`ModelRegistry.load_model()`)
   ```
   → Creates ModelHelper instance
   → ModelHelper loads tokenizer
   → ModelHelper loads model with quantization
   → ModelHelper wraps each layer with BlockOutputWrapper
   → Stores in registry.models dict
   ```

3. **Data Loading** (`load_prompt_pairs()`)
   ```
   → Reads base_prompts.json
   → Reads counterfactual_prompts.json
   → Matches by prompt_id
   → Returns list of paired prompts
   ```

4. **Analysis Execution**
   
   **Causal Tracing:**
   ```
   CausalTracer.__init__()
   → Stores model reference
   
   For each prompt_pair:
       tracer.get_baseline_logits(original)
       tracer.get_hidden_states_manual(counterfactual)
       
       For each layer:
           tracer.get_logits_with_patch()
           → Unwraps layers
           → Forwards through model
           → Patches at target layer
           → Computes logits
           → Calculates causal score
       
       Saves to DataFrame
   ```
   
   **Direct Logit Attribution:**
   ```
   DirectLogitAttribution.__init__()
   
   For selected layers:
       dla._forward_to_layer(target_layer)
       
       # Isolate attention
       attn_output = layer.self_attn(hidden_states)
       dla._project_to_vocab(attn_output, token_id)
       
       # Isolate MLP
       mlp_output = layer.mlp(hidden_states)
       dla._project_to_vocab(mlp_output, token_id)
       
       Saves contributions
   ```
   
   **Tuned Lens:**
   ```
   TunedLensAnalyzer.__init__()
   
   For each prompt:
       hidden = embed_tokens(prompt)
       
       For each layer:
           hidden = layer(hidden)
           logits = lm_head(norm(hidden))
           top_k = logits.topk(k=10)
           
           Record predictions
   ```

5. **Visualization** (called after each analysis)
   ```
   → Loads result DataFrames
   → Creates matplotlib figures
   → Generates subplots
   → Saves as PNG files
   → Displays (if interactive)
   ```

---

## Understanding the Output

### Causal Tracing Output

**Files:**
- `results/causal_tracing/detailed_layer_data_{model}.csv`
- `results/causal_tracing/summary_results_{model}.csv`  
- `results/causal_tracing/causal_tracing_{model}.png`

**Key Columns in `summary_results`:**

| Column | Meaning |
|--------|---------|
| `peak_layer` | Layer with highest causal score (most important) |
| `num_choice_flips` | How many layers flip the model's decision |
| `gini_coefficient` | How concentrated causal effects are (0=equal, 1=concentrated) |
| `top5_concentration` | Fraction of total effect from top 5 layers |
| `chose_ethical_base` | Whether model chose the ethical option |

**Interpreting the Plot:**

```
Average Causal Score
     │     ╱╲
     │    ╱  ╲      <- Peak around layer 25
     │   ╱    ╲       (This layer is critical!)
     │  ╱      ╲
     │ ╱        ╲___
     │╱
─────┼─────────────────→ Layer
     0  10  20  30  40
```

- **High scores** → Layer is causally important
- **Peak location** → Where decision is made (early vs late)
- **Width of peak** → Whether decision is localized or distributed

### Direct Logit Attribution Output

**Files:**
- `results/attribution/attribution_{model}.csv`
- `results/attribution/attribution_{model}.png`

**Key Columns:**

| Column | Meaning |
|--------|---------|
| `attn_to_1` | How much attention heads push toward option 1 |
| `mlp_to_1` | How much MLPs push toward option 1 |
| `total_to_1` | Combined contribution to option 1 |
| `attn_mlp_ratio` | Attention importance (0=MLP dominant, 1=Attention dominant) |

**Interpreting the Plots:**

- **Component Contributions:** Shows attention vs MLP across layers
- **Decision Differential:** Positive = prefers option 1, Negative = prefers option 2
- **Heatmap:** Color intensity shows contribution magnitude
- **Ratio Plot:** Shows whether attention or MLPs dominate decision-making

### Tuned Lens Output

**Files:**
- `results/tuned_lens/tuned_lens_{model}.json`
- `results/tuned_lens/token_trajectories_{model}.csv`
- `results/tuned_lens/tuned_lens_{model}.png`

**JSON Structure:**
```json
[
  {
    "layer": 0,
    "prompt_id": "scenario_001",
    "top_tokens": [" the", " a", " to", ...],
    "top_probs": [0.15, 0.12, 0.08, ...],
    "top_logits": [5.2, 4.8, 4.1, ...]
  }
]
```

**Interpreting the Trajectory Plot:**

```
Logit Value
     │     Option 1 ─────╱
     │                  ╱
     │                 ╱
     │               ╱
     │    Option 2 ╱
     │         ___╱
─────┼─────────────────→ Layer
     0  10  20  30  40
```

- **Crossing point** → Layer where model "changes its mind"
- **Early divergence** → Decision made in early layers
- **Late divergence** → Decision made in late layers

---

## Advanced Usage

### Analyzing Custom Models

Add your model to `config/models.json`:

```json
{
  "my_custom_model": {
    "name": "username/model-name-on-huggingface",
    "token": "hf_YourToken"
  }
}
```

Then load it:

```python
registry.load_model(
    'my_custom_model',
    MODELS['my_custom_model']['name'],
    MODELS['my_custom_model']['token'],
    use_4bit=True  # Set based on model size
)
```

### Memory Management for Large Models

```python
# Load model with 4-bit quantization
registry.load_model('llama', ..., use_4bit=True)

# Run analysis
results = run_batch_causal_tracing(...)

# Unload when done
registry.unload_model('llama')

# Check memory usage
memory_info = registry.get_memory_info()
print(memory_info)
```

### Processing Prompts in Batches

```python
import torch

# Process in chunks of 10 to save memory
chunk_size = 10
all_results = []

for i in range(0, len(prompt_pairs), chunk_size):
    chunk = prompt_pairs[i:i+chunk_size]
    
    df_layers, df_summary = run_batch_causal_tracing(
        'gpt2', chunk, registry,
        output_dir=f'results/batch_{i}'
    )
    
    all_results.append(df_summary)
    
    # Clear GPU cache between batches
    torch.cuda.empty_cache()

# Combine results
import pandas as pd
final_results = pd.concat(all_results, ignore_index=True)
```

### Comparing Multiple Models

```python
models_to_compare = ['gpt2', 'llama', 'mixtral']
results = {}

for model_key in models_to_compare:
    registry.load_model(model_key, ...)
    
    df_layers, df_summary = run_batch_causal_tracing(
        model_key, prompt_pairs, registry
    )
    
    results[model_key] = df_summary
    registry.unload_model(model_key)

# Compare peak layers across models
for model, df in results.items():
    avg_peak = df['peak_layer'].mean()
    print(f"{model}: avg peak layer = {avg_peak:.1f}")
```

### Custom Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv('results/causal_tracing/summary_results_gpt2.csv')

# Custom plot
plt.figure(figsize=(10, 6))
sns.histplot(df['peak_layer'], bins=12, kde=True)
plt.title('Distribution of Peak Causal Layers')
plt.xlabel('Layer')
plt.ylabel('Count')
plt.savefig('my_custom_plot.png', dpi=300)
plt.show()
```

---

## Troubleshooting

### Common Issues

#### 1. "CUDA out of memory"

**Solution:** Use 4-bit quantization or smaller batch size

```python
# Enable quantization
registry.load_model('llama', ..., use_4bit=True)

# Or process fewer prompts at once
prompt_pairs = load_prompt_pairs()[:50]
```

#### 2. "Cannot find distinct token IDs for '1' and '2'"

**Solution:** The tokenizer may not separate digits. Check tokenization:

```python
from src.utils.tokenizer_utils import analyze_tokenization

model_helper = registry.get_model('gpt2')
info = analyze_tokenization(model_helper.tokenizer, "Answer: 1")
print(info)
```

If digits share token IDs, modify your prompts to use words: "Option A" vs "Option B"

#### 3. "Model not found in config/models.json"

**Solution:** Ensure your `config/models.json` exists and is valid JSON

```bash
# Check if file exists
ls config/models.json

# Validate JSON
python -m json.tool config/models.json
```

#### 4. "ModuleNotFoundError: No module named 'src'"

**Solution:** Either install the package or add to Python path

```bash
# Option 1: Install package
pip install -e .

# Option 2: Run from project root
cd /path/to/mechanistic-interpretability
python scripts/run_single_model.py
```

#### 5. Slow Execution

**Optimization tips:**
- Use GPU if available (automatically detected)
- Enable 4-bit quantization for large models
- Reduce number of prompts for testing
- Process in batches with cache clearing

```python
# Check if CUDA is available
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```


### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests if applicable
5. Run tests and linting
6. Commit: `git commit -m "Add feature X"`
7. Push: `git push origin feature-name`
8. Open a Pull Request

---

## Additional Resources

### Papers

- **Causal Tracing:** [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (Meng et al., 2022)
- **Logit Attribution:** [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (Geva et al., 2021)
- **Tuned Lens:** [Eliciting Latent Predictions from Transformers](https://arxiv.org/abs/2303.08112) (Belrose et al., 2023)

