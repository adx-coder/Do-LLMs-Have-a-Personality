import torch
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm


class TunedLensAnalyzer:
    
    def __init__(self, model_helper):
        self.model_helper = model_helper
        self.model = model_helper.model
        self.tokenizer = model_helper.tokenizer
        self.layers = self.model.model.layers
        self.norm = self.model.model.norm

    def analyze_prompt(self, prompt: str, top_k: int = 10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_ids = inputs['input_ids']

        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(input_ids)
            layer_predictions = []

            attention_mask, position_ids, position_embeddings = \
                self._prepare_layer_args(inputs, hidden_states)

            for layer_idx in range(len(self.layers)):
                layer = self.layers[layer_idx]
                if hasattr(layer, 'block'):
                    layer = layer.block

                layer_kwargs = {
                    'attention_mask': attention_mask, 
                    'position_ids': position_ids
                }
                if position_embeddings is not None:
                    layer_kwargs['position_embeddings'] = position_embeddings

                layer_output = layer(hidden_states, **layer_kwargs)
                hidden_states = layer_output[0] if isinstance(layer_output, tuple) \
                    else layer_output

                # Project to vocabulary
                normed = self.norm(hidden_states)
                logits = self.model.lm_head(normed)

                # Get top-k
                last_token_logits = logits[0, -1, :]
                top_k_logits, top_k_indices = torch.topk(last_token_logits, top_k)

                top_k_tokens = [
                    self.tokenizer.decode([idx]) for idx in top_k_indices
                ]
                top_k_probs = torch.softmax(top_k_logits, dim=0).cpu().numpy()

                layer_predictions.append({
                    'layer': layer_idx,
                    'top_tokens': top_k_tokens,
                    'top_probs': top_k_probs.tolist(),
                    'top_logits': top_k_logits.cpu().numpy().tolist()
                })

        return layer_predictions

    def get_token_trajectory(self, prompt: str, target_tokens: list):
        # Get token IDs
        target_token_ids = []
        for token_str in target_tokens:
            full_enc = self.tokenizer.encode(
                f"Answer: {token_str}", add_special_tokens=False
            )
            target_token_ids.append(full_enc[-1])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        trajectories = {token: [] for token in target_tokens}

        with torch.no_grad():
            hidden_states = self.model.model.embed_tokens(inputs['input_ids'])
            attention_mask, position_ids, position_embeddings = \
                self._prepare_layer_args(inputs, hidden_states)

            for layer_idx in range(len(self.layers)):
                layer = self.layers[layer_idx]
                if hasattr(layer, 'block'):
                    layer = layer.block

                layer_kwargs = {
                    'attention_mask': attention_mask, 
                    'position_ids': position_ids
                }
                if position_embeddings is not None:
                    layer_kwargs['position_embeddings'] = position_embeddings

                layer_output = layer(hidden_states, **layer_kwargs)
                hidden_states = layer_output[0] if isinstance(layer_output, tuple) \
                    else layer_output

                normed = self.norm(hidden_states)
                logits = self.model.lm_head(normed)
                last_logits = logits[0, -1, :]

                for token_str, token_id in zip(target_tokens, target_token_ids):
                    trajectories[token_str].append(last_logits[token_id].item())

        df = pd.DataFrame(trajectories)
        df['layer'] = range(len(self.layers))
        return df

    def _prepare_layer_args(self, inputs, hidden_states):
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=self.model.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * \
                torch.finfo(hidden_states.dtype).min

        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            position_embeddings = self.model.model.rotary_emb(
                hidden_states, position_ids
            )

        return attention_mask, position_ids, position_embeddings


def run_tuned_lens_analysis(model_key, prompt_pairs, registry,
                           num_samples=10, top_k=15, 
                           output_dir='results/tuned_lens'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nTUNED LENS ANALYSIS - {model_key}\n")

    model_helper = registry.get_model(model_key)
    lens = TunedLensAnalyzer(model_helper)

    all_predictions = []
    trajectories = []

    for pair in tqdm(prompt_pairs[:num_samples], desc="Analyzing"):
        try:
            predictions = lens.analyze_prompt(pair['base_prompt'], top_k=top_k)
            for pred in predictions:
                pred['prompt_id'] = pair['prompt_id']
                all_predictions.append(pred)

            traj = lens.get_token_trajectory(pair['base_prompt'], ['1', '2'])
            traj['prompt_id'] = pair['prompt_id']
            trajectories.append(traj)
        except Exception as e:
            print(f"Error: {e}")
            continue

    with open(output_path / f'tuned_lens_{model_key}.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)

    if trajectories:
        df_traj = pd.concat(trajectories, ignore_index=True)
        df_traj.to_csv(
            output_path / f'token_trajectories_{model_key}.csv', index=False
        )

    print(f"\nResults saved to: {output_path}")
    return all_predictions, trajectories