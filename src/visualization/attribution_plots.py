import matplotlib.pyplot as plt
from pathlib import Path


def visualize_attribution(df_attribution, model_key, 
                         output_dir='results/attribution'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Direct Logit Attribution - {model_key}', 
                 fontsize=16, fontweight='bold')

    # Plot 1: Component Contributions to "1"
    ax1 = axes[0, 0]
    if 'attn_to_1' in df_attribution.columns and 'mlp_to_1' in df_attribution.columns:
        avg_by_layer = df_attribution.groupby('layer').agg({
            'attn_to_1': 'mean',
            'mlp_to_1': 'mean',
            'total_to_1': 'mean'
        })
        ax1.plot(avg_by_layer.index, avg_by_layer['attn_to_1'],
                label='Attention → "1"', marker='o', linewidth=2)
        ax1.plot(avg_by_layer.index, avg_by_layer['mlp_to_1'],
                label='MLP → "1"', marker='s', linewidth=2)
        ax1.plot(avg_by_layer.index, avg_by_layer['total_to_1'],
                label='Total → "1"', marker='^', linewidth=2, linestyle='--')
        ax1.set_xlabel('Layer', fontweight='bold')
        ax1.set_ylabel('Contribution', fontweight='bold')
        ax1.set_title('Component Contributions to "1"', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Plot 2: Decision Differential
    ax2 = axes[0, 1]
    if 'total_to_1' in df_attribution.columns and 'total_to_2' in df_attribution.columns:
        df_attribution['diff'] = df_attribution['total_to_1'] - df_attribution['total_to_2']
        avg_diff = df_attribution.groupby('layer')['diff'].mean()

        ax2.bar(avg_diff.index, avg_diff.values, 
                alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer', fontweight='bold')
        ax2.set_ylabel('Contribution Diff ("1" - "2")', fontweight='bold')
        ax2.set_title('Layer-wise Decision Preference', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Heatmap
    ax3 = axes[1, 0]
    contrib_cols = [c for c in df_attribution.columns if '_to_' in c and c != 'diff']
    if contrib_cols and len(df_attribution) > 0:
        heatmap_data = df_attribution.groupby('layer')[contrib_cols].mean()
        im = ax3.imshow(heatmap_data.T, aspect='auto', cmap='RdBu_r',
                       interpolation='nearest')

        ax3.set_xticks(range(len(heatmap_data.index)))
        ax3.set_xticklabels(heatmap_data.index)
        ax3.set_yticks(range(len(contrib_cols)))
        ax3.set_yticklabels(contrib_cols, fontsize=9)
        ax3.set_xlabel('Layer', fontweight='bold')
        ax3.set_title('Contribution Heatmap', fontweight='bold')
        plt.colorbar(im, ax=ax3, label='Contribution')

    # Plot 4: Attention vs MLP Ratio
    ax4 = axes[1, 1]
    if 'attn_to_1' in df_attribution.columns and 'mlp_to_1' in df_attribution.columns:
        df_attribution['attn_mlp_ratio'] = (
            abs(df_attribution['attn_to_1']) /
            (abs(df_attribution['attn_to_1']) + abs(df_attribution['mlp_to_1']) + 1e-8)
        )
        ratio_by_layer = df_attribution.groupby('layer')['attn_mlp_ratio'].mean()

        ax4.plot(ratio_by_layer.index, ratio_by_layer.values,
                marker='o', linewidth=2, color='purple')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal split')
        ax4.fill_between(ratio_by_layer.index, 0.5, ratio_by_layer.values,
                        where=(ratio_by_layer.values > 0.5), 
                        alpha=0.3, color='blue', label='Attention-dominant')
        ax4.fill_between(ratio_by_layer.index, 0.5, ratio_by_layer.values,
                        where=(ratio_by_layer.values < 0.5), 
                        alpha=0.3, color='orange', label='MLP-dominant')
        ax4.set_xlabel('Layer', fontweight='bold')
        ax4.set_ylabel('Attention / (Attention + MLP)', fontweight='bold')
        ax4.set_title('Component Importance Ratio', fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_path / f'attribution_{model_key}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_path}")
    plt.show()