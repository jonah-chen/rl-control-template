import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.stats as sp
import seaborn as sns

def analyze_results(base_dir):
    results = []

    # Dynamically find all unique seed directories
    # Assuming seed directories are always a single number and directly under lr=...,bs=...
    all_seed_dirs = glob.glob(os.path.join(base_dir, "hidden=*,layers=*", "lr=*,bs=*", "*"))
    # Extract seed numbers and convert to integers
    unique_seeds = sorted(list(set(int(os.path.basename(s)) for s in all_seed_dirs if os.path.basename(s).isdigit())))

    # Walk through the directory structure
    # Expected structure: base_dir / hidden=...,layers=... / lr=...,bs=... / seed / eval_returns.txt

    # Get all first level directories (hidden/layers)
    layer_dirs = glob.glob(os.path.join(base_dir, "hidden=*,layers=*"))

    for layer_dir in layer_dirs:
        layer_dir_name = os.path.basename(layer_dir)
        # Parse hidden and layers
        params = dict(item.split("=") for item in layer_dir_name.split(","))
        hidden_str = params.get("hidden")
        layers_str = params.get("layers")

        if hidden_str is None or layers_str is None:
            continue

        hidden = int(hidden_str)
        layers = int(layers_str)

        # Get all second level directories (lr/bs)
        lr_bs_dirs = glob.glob(os.path.join(layer_dir, "lr=*,bs=*"))

        for lr_bs_dir in lr_bs_dirs:
            lr_bs_dir_name = os.path.basename(lr_bs_dir)
            # Parse lr and bs
            params = dict(item.split("=") for item in lr_bs_dir_name.split(","))
            lr_str = params.get("lr")
            bs_str = params.get("bs")

            if lr_str is None or bs_str is None:
                continue

            lr = float(lr_str)
            bs = int(bs_str)

            all_returns = []
            failures_per_seed = []

            # Iterate through dynamically found seeds
            for seed in unique_seeds:
                seed_dir = os.path.join(lr_bs_dir, str(seed))
                eval_file = os.path.join(seed_dir, "metrics.csv")
                metrics = np.loadtxt(eval_file, skiprows=1, delimiter=',')
                step = metrics[:,0]
                returns = metrics[:,1]
                returns = returns[step < 195000]
                step = step[step < 195000]
                # add the step = 195000 point with the last return value
                step = np.append(step, 195000)
                returns = np.append(returns, returns[-1])
                auc = np.trapezoid(returns, step) / 195000
                # Read lines, convert to float
                all_returns.append(auc)

            failures_per_seed.sort()
            failures_per_seed_str = ", ".join(map(str, failures_per_seed))

            # Process results for this configuration
            if all_returns:
                total_count = len(all_returns)
                # Count failures
                failure_count = 0

                avg_return = np.mean(all_returns)
                # Check for constant values to avoid bootstrap errors/warnings
                if np.min(all_returns) == np.max(all_returns):
                    std_return = 0.0
                else:
                    res = sp.bootstrap((all_returns,), np.mean)
                    std_return = res.confidence_interval.high - avg_return
                    if np.isnan(std_return):
                        std_return = np.nan

                # Unfiltered is now the same as average_return
                avg_return_unfiltered = avg_return
                std_return_unfiltered = std_return

                results.append({
                    "hidden": hidden,
                    "layers": layers,
                    "lr": lr,
                    "bs": bs,
                    "average_return": avg_return,
                    "std_return": std_return,
                    "average_return_unfiltered": avg_return_unfiltered,
                    "std_return_unfiltered": std_return_unfiltered,
                    "failure_count": failure_count,
                    "failures_per_seed": failures_per_seed_str,
                    "total_samples": total_count
                })

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort for better readability
    if not df.empty:
        df = df.sort_values(by=["hidden", "layers", "lr", "bs"])

        # Print the table
        print(df.to_markdown(index=False))

        # Also save to CSV for reference
        df.to_csv("analysis_results.csv", index=False)
        print("\nResults saved to analysis_results.csv")

        # Generate Heatmaps
        generate_heatmaps(df)
    else:
        print("No results found.")

def generate_heatmaps(df):
    # Get unique hidden and layers combinations
    configs = df[['hidden', 'layers']].drop_duplicates().values

    for hidden, layers in configs:
        subset = df[(df['hidden'] == hidden) & (df['layers'] == layers)]

        if subset.empty:
            continue

        # Create figure with one subplot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f'Hidden: {hidden}, Layers: {layers}', fontsize=16)

        # Plot 1: Average Return (Filtered)
        pivot_filtered = subset.pivot(index='bs', columns='lr', values='average_return')
        pivot_std_filtered = subset.pivot(index='bs', columns='lr', values='std_return')

        # Create annotation matrix for Plot 1
        annot_filtered = pivot_filtered.copy().astype(object)
        for r in pivot_filtered.index:
            for c in pivot_filtered.columns:
                val = pivot_filtered.loc[r, c]
                std = pivot_std_filtered.loc[r, c]
                if pd.notna(val):
                    annot_filtered.loc[r, c] = f"{val:.1f}\n({std:.1f})"
                else:
                    annot_filtered.loc[r, c] = ""

        sns.heatmap(pivot_filtered, annot=annot_filtered, fmt="", cmap="Greens", ax=ax)
        ax.set_title('Average Return (Raw)\nMean\n(Std)')
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Batch Size')

        plt.tight_layout()
        filename = f"heatmap_hidden{hidden}_layers{layers}.png"
        plt.savefig(filename)
        print(f"Saved heatmap to {filename}")
        plt.close()
    
if __name__ == "__main__":
    base_directory = "multirun/2025-12-11/00-22-08"  # Change this to your results directory
    analyze_results(base_directory)