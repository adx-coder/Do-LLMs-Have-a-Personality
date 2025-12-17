import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def visualize_tuned_lens(trajectories, model_key, output_dir='results/tuned_lens'):
    if not trajectories:
        print("No trajectory data")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df_all = pd.concat(trajectories, ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Tuned Lens - {model_key}', fontsize=16, fontweight='bold')

    ax1 = axes[0]
    avg_traj = df_all.groupby('layer')[['1', '2']].mean()
    ax1.plot(avg_traj.index, avg_traj['1'], label='Token "1"', marker='o')
    ax1.plot(avg_traj.index, avg_traj['2'], label='Token "2"', marker='s')
    ax1.set_title('Token Logit Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    df_all['margin'] = df_all['1'] - df_all['2']
    avg_margin = df_all.groupby('layer')['margin'].mean()
    ax2.plot(avg_margin.index, avg_margin.values, marker='o', color='green')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_title('Decision Margin Evolution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f'tuned_lens_{model_key}.png', dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_path / f'tuned_lens_{model_key}.png'}")
    plt.show()