import os
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Constant to replace -5000 (failure) in unfiltered average calculation
FAILURE_REPLACEMENT = -3000

def analyze_results(base_dir):
    results = []
    
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
            
            # Iterate through seeds 1 to 5
            for seed in range(1, 6):
                seed_dir = os.path.join(lr_bs_dir, str(seed))
                eval_file = os.path.join(seed_dir, "eval_returns.txt")
                
                current_seed_failures = 0
                if os.path.exists(eval_file):
                    try:
                        with open(eval_file, 'r') as f:
                            # Read lines, convert to float
                            seed_returns = [float(line.strip()) for line in f if line.strip()]
                            all_returns.extend(seed_returns)
                            current_seed_failures = sum(1 for r in seed_returns if r <= -5000)
                    except Exception as e:
                        print(f"Error reading {eval_file}: {e}")
                else:
                    print(f"Warning: {eval_file} not found")
                
                failures_per_seed.append(current_seed_failures)
            
            failures_per_seed.sort()
            failures_per_seed_str = ", ".join(map(str, failures_per_seed))
            
            # Process results for this configuration
            if all_returns:
                total_count = len(all_returns)
                # Filter out -5000
                valid_returns = [r for r in all_returns if r > -5000]
                failure_count = total_count - len(valid_returns)
                
                if valid_returns:
                    avg_return = np.mean(valid_returns)
                    std_return = np.std(valid_returns)
                else:
                    avg_return = np.nan # Or some other indicator if all failed
                    std_return = np.nan
                
                # Calculate unfiltered average with adjustment
                # Replace failures (-5000) with FAILURE_REPLACEMENT
                adjusted_returns = [r if r > -5000 else FAILURE_REPLACEMENT for r in all_returns]
                avg_return_unfiltered = np.mean(adjusted_returns)
                std_return_unfiltered = np.std(adjusted_returns)

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
            
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Hidden: {hidden}, Layers: {layers}', fontsize=16)
        
        # Plot 1: Average Return (Filtered)
        pivot_filtered = subset.pivot(index='bs', columns='lr', values='average_return')
        pivot_std_filtered = subset.pivot(index='bs', columns='lr', values='std_return')
        
        # Create annotation matrix for Plot 1
        annot_filtered = pivot_filtered.applymap(lambda x: f"{x:.0f}") + "\n(" + pivot_std_filtered.applymap(lambda x: f"{x:.0f}") + ")"
        
        sns.heatmap(pivot_filtered, annot=annot_filtered, fmt="", cmap="viridis", ax=axes[0])
        axes[0].set_title('Average Return (Filtered > -5000)\nMean\n(Std)')
        axes[0].set_xlabel('Learning Rate')
        axes[0].set_ylabel('Batch Size')
        
        # Plot 2: Failure Count
        pivot_failure = subset.pivot(index='bs', columns='lr', values='failure_count')
        pivot_failures_per_seed = subset.pivot(index='bs', columns='lr', values='failures_per_seed')
        
        # Create annotation matrix for Plot 2
        annot_failure = pivot_failure.applymap(lambda x: f"{x:.0f}") + "\n(" + pivot_failures_per_seed.astype(str) + ")"
        
        sns.heatmap(pivot_failure, annot=annot_failure, fmt="", cmap="Reds", ax=axes[1])
        axes[1].set_title('Failure Count (out of 250)\nTotal\n(Per Seed Sorted)')
        axes[1].set_xlabel('Learning Rate')
        axes[1].set_ylabel('Batch Size')
        
        plt.tight_layout()
        filename = f"heatmap_hidden{hidden}_layers{layers}.png"
        plt.savefig(filename)
        print(f"Saved heatmap to {filename}")
        plt.close()


if __name__ == "__main__":
    base_dir = "multirun/2025-12-08/07-00-36"
    analyze_results(base_dir)
