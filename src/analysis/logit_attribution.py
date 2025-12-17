import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm


class DirectLogitAttribution:
    
    def __init__(self, model_helper):
        self.model_helper = model_helper
        self.model = model_helper.model
        self.tokenizer = model_helper.tokenizer
        self.layers = self.model.model.layers
        self.norm = self.model.model.norm

    def compute_component_contributions(self, prompt: str, target_layer: int,
                                       choice_tokens: list, choice_labels: list):
        # compute attention and MLP contributions for a specific layer
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            # forward to target layer
            hidden_states = self._forward_to_layer(inputs, target_layer)
            hidden_before = hidden_states.clone()

            # get unwrapped layer
            layer = self.layers[target_layer]
            if hasattr(layer, 'block'):
                layer = layer.block

            # prepare layer inputs
            attention_mask, position_ids, position_embeddings = self._prepare_layer_args(inputs, hidden_states)

            # ATTENTION CONTRIBUTION
            attn_contributions = {}
            try:
                if hasattr(layer, 'input_layernorm'):
                    normed = layer.input_layernorm(hidden_states)
                else:
                    normed = hidden_states

                layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
                if position_embeddings is not None:
                    layer_kwargs['position_embeddings'] = position_embeddings

                if hasattr(layer, 'self_attn'):
                    attn_output = layer.self_attn(normed, **layer_kwargs)[0]

                    for token_id, label in zip(choice_tokens, choice_labels):
                        contribution = self._project_to_vocab(attn_output[:, -1, :], token_id)
                        attn_contributions[f'attn_to_{label}'] = contribution
            except Exception as e:
                print(f"Attention analysis failed: {e}")

            # MLP CONTRIBUTION
            mlp_contributions = {}
            try:
                hidden_after_attn = hidden_before + attn_output if 'attn_output' in locals() else hidden_before

                if hasattr(layer, 'post_attention_layernorm'):
                    mlp_input = layer.post_attention_layernorm(hidden_after_attn)
                else:
                    mlp_input = hidden_after_attn

                if hasattr(layer, 'mlp'):
                    mlp_output = layer.mlp(mlp_input)

                    for token_id, label in zip(choice_tokens, choice_labels):
                        contribution = self._project_to_vocab(mlp_output[:, -1, :], token_id)
                        mlp_contributions[f'mlp_to_{label}'] = contribution
            except Exception as e:
                print(f"MLP analysis failed: {e}")

            # TOTAL CONTRIBUTION
            total_contributions = {}
            try:
                layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
                if position_embeddings is not None:
                    layer_kwargs['position_embeddings'] = position_embeddings

                layer_output = layer(hidden_before, **layer_kwargs)
                layer_output = layer_output[0] if isinstance(layer_output, tuple) else layer_output
                delta_h = layer_output - hidden_before

                for token_id, label in zip(choice_tokens, choice_labels):
                    contribution = self._project_to_vocab(delta_h[:, -1, :], token_id)
                    total_contributions[f'total_to_{label}'] = contribution
            except Exception as e:
                print(f"Total contribution failed: {e}")

        return {
            'layer': target_layer,
            'head_contributions': attn_contributions,
            'mlp_contributions': mlp_contributions,
            'total_contributions': total_contributions,
            'prompt': prompt
        }

    def _forward_to_layer(self, inputs, target_layer):
        # forward through layers up to target
        input_ids = inputs['input_ids']
        hidden_states = self.model.model.embed_tokens(input_ids)

        for i in range(target_layer):
            layer = self.layers[i]
            if hasattr(layer, 'block'):
                layer = layer.block

            attention_mask, position_ids, position_embeddings = self._prepare_layer_args(inputs, hidden_states)
            layer_kwargs = {'attention_mask': attention_mask, 'position_ids': position_ids}
            if position_embeddings is not None:
                layer_kwargs['position_embeddings'] = position_embeddings

            layer_output = layer(hidden_states, **layer_kwargs)
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        return hidden_states

    def _prepare_layer_args(self, inputs, hidden_states):
        ## prepare arguments for layer forward pass
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)

        batch_size, seq_length = input_ids.shape
        position_ids = torch.arange(seq_length, dtype=torch.long, device=self.model.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min

        position_embeddings = None
        if hasattr(self.model.model, 'rotary_emb'):
            position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        return attention_mask, position_ids, position_embeddings

    def _project_to_vocab(self, hidden_state, token_id):
        ## project hidden state to vocabulary token
        W_U = self.model.lm_head.weight
        token_direction = W_U[token_id]
        contribution = torch.dot(hidden_state.squeeze(), token_direction).item()
        return contribution


def run_attribution_analysis(model_key, prompt_pairs, registry,
                            target_layers=None, output_dir='results/attribution'):
    ## run DLA analysis
    from ..utils.tokenizer_utils import get_choice_tokens_robust
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDIRECT LOGIT ATTRIBUTION - {model_key}\n")

    model_helper = registry.get_model(model_key)
    dla = DirectLogitAttribution(model_helper)
    token_1, token_2 = get_choice_tokens_robust(model_helper.tokenizer)

    if target_layers is None:
        num_layers = len(dla.layers)
        target_layers = [num_layers // 2, int(num_layers * 0.7), num_layers - 3, num_layers - 2, num_layers - 1]

    print(f"Target layers: {target_layers}\n")

    all_results = []
    for pair in tqdm(prompt_pairs[:20], desc="Analyzing attributions"):
        for layer_idx in target_layers:
            try:
                result = dla.compute_component_contributions(
                    prompt=pair['base_prompt'],
                    target_layer=layer_idx,
                    choice_tokens=[token_1, token_2],
                    choice_labels=['1', '2']
                )

                row = {'prompt_id': pair['prompt_id'], 'layer': layer_idx}
                row.update(result['head_contributions'])
                row.update(result['mlp_contributions'])
                row.update(result['total_contributions'])
                all_results.append(row)
            except Exception as e:
                print(f"Error: {e}")
                continue

    df = pd.DataFrame(all_results)
    df.to_csv(output_path / f'attribution_{model_key}.csv', index=False)
    print(f"\nResults saved to: {output_path}")
    return df