import matplotlib.pyplot as plt
from pathlib import Path


def visualize_causal_tracing(df_layers, df_summary, model_key, 
                            output_dir='results/causal_tracing'):
    """Create visualizations for causal tracing results."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    avg_by_layer = df_layers.groupby('Layer')['Causal_Score'].agg([
        'mean', 'std', 'median'
    ])

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Average causal score
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(avg_by_layer.index, avg_by_layer['mean'], 
             linewidth=2.5, color='#2E86AB')
    ax1.fill_between(
        avg_by_layer.index, 
        avg_by_layer['mean'] - avg_by_layer['std'],
        avg_by_layer['mean'] + avg_by_layer['std'], 
        alpha=0.3, color='#2E86AB'
    )
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Layer', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Causal Score', fontsize=13, fontweight='bold')
    ax1.set_title(f'Causal Contribution - {model_key}', 
                  fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.4)

    # Plot 2: Peak layers
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df_summary['peak_layer'], bins=30, 
             color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_title('Peak Layers', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Layer Index', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.grid(True, alpha=0.4, axis='y')

    # Plot 3: Choice flips
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(
        df_summary['num_choice_flips'],
        bins=range(0, int(df_summary['num_choice_flips'].max())+2),
        color='#F18F01', alpha=0.7, edgecolor='black'
    )
    ax3.set_title('Choice Flips', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Number of Flips', fontweight='bold')
    ax3.set_ylabel('Frequency', fontweight='bold')
    ax3.grid(True, alpha=0.4, axis='y')

    # Plot 4: Ethical alignment
    ax4 = fig.add_subplot(gs[1, 1])
    ethical_counts = df_summary['chose_ethical_base'].value_counts()
    ax4.pie(
        ethical_counts, 
        labels=['Ethical', 'Unethical'], 
        autopct='%1.1f%%',
        colors=['#27AE60', '#E74C3C'], 
        startangle=90
    )
    ax4.set_title('Ethical Alignment', fontsize=13, fontweight='bold')

    # Plot 5: Concentration metrics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(df_summary['gini_coefficient'], bins=20, 
             color='#9B59B6', alpha=0.7, edgecolor='black')
    ax5.set_title('Gini Coefficient Distribution', 
                  fontsize=13, fontweight='bold')
    ax5.set_xlabel('Gini Coefficient', fontweight='bold')
    ax5.set_ylabel('Frequency', fontweight='bold')
    ax5.grid(True, alpha=0.4, axis='y')

    # Plot 6: Temporal analysis
    ax6 = fig.add_subplot(gs[2, :])
    temporal_means = df_summary[['early_mean', 'middle_mean', 'late_mean']].mean()
    temporal_std = df_summary[['early_mean', 'middle_mean', 'late_mean']].std()
    
    positions = ['Early\n(0-33%)', 'Middle\n(33-66%)', 'Late\n(66-100%)']
    ax6.bar(positions, temporal_means, yerr=temporal_std, 
            capsize=5, color=['#3498DB', '#E67E22', '#E74C3C'], 
            alpha=0.7, edgecolor='black')
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax6.set_ylabel('Average Causal Score', fontsize=13, fontweight='bold')
    ax6.set_title('Temporal Distribution of Causal Effects', 
                  fontsize=15, fontweight='bold')
    ax6.grid(True, alpha=0.4, axis='y')

    plt.tight_layout()
    save_path = output_path / f'causal_tracing_{model_key}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {save_path}")
    plt.show()
    
    return avg_by_layer