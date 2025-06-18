import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import io
from scipy import stats
import seaborn as sns

def create_bias_variance_plot(group_data, minibatch_idx, parameter_name):
    """
    Create a plot showing the relationship between (1/k - 1/8) and gradient residual norm.
    
    Args:
        group_data: DataFrame with data for a specific minibatch and parameter
        minibatch_idx: The minibatch index
        parameter_name: Name of the parameter
    
    Returns:
        PIL Image of the plot
    """
    # Calculate 1/k - 1/8 for each subset size
    subset_sizes = group_data['subset_size'].values
    x_values = 1/subset_sizes - 1/8
    y_values = group_data['residual_norm'].values
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(x_values, y_values, alpha=0.6, s=20, color='blue')
    
    # Line of best fit
    if len(x_values) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
        line_x = np.linspace(min(x_values), max(x_values), 100)
        line_y = slope * line_x + intercept
        plt.plot(line_x, line_y, 'r-', alpha=0.8, linewidth=2, 
                label=f'Slope: {slope:.3e}, R²: {r_value**2:.3f}')
        plt.legend()
    else:
        slope = np.nan
    
    # Formatting
    plt.xlabel('1/k - 1/8', fontsize=12)
    plt.ylabel('||Gradient Residual||_F', fontsize=12)
    plt.title(f'Bias-Variance Analysis\nMinibatch: {minibatch_idx}, Parameter: {parameter_name}\nSlope: {slope:.3e}', 
              fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def process_bias_variance_csv(csv_path, output_dir=None):
    """
    Process bias/variance CSV file and create plots for each parameter at each minibatch.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
        output_dir: Directory to save plots (optional)
    
    Returns:
        Dictionary mapping (minibatch_idx, parameter_name) to PIL Images
    """
    df = pd.read_csv(csv_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    plots = {}
    
    # Group by minibatch and parameter
    grouped = df.groupby(['minibatch_idx', 'parameter_name'])
    
    print(f"Processing {len(grouped)} parameter-minibatch combinations...")
    
    for (minibatch_idx, parameter_name), group in grouped:
        try:
            # Create plot
            img = create_bias_variance_plot(group, minibatch_idx, parameter_name)
            plots[(minibatch_idx, parameter_name)] = img
            
            # Save individual plot if output directory specified
            if output_dir:
                filename = f"minibatch_{minibatch_idx:06d}_{parameter_name.replace('/', '_')}.png"
                img.save(output_dir / filename)
                
        except Exception as e:
            print(f"Error processing minibatch {minibatch_idx}, parameter {parameter_name}: {e}")
    
    return plots

def create_gif_from_plots(plots, output_path, duration=500, parameter_filter=None):
    """
    Create a GIF from the plots, cycling through minibatches for each parameter.
    
    Args:
        plots: Dictionary of plots from process_bias_variance_csv
        output_path: Path to save the GIF
        duration: Duration per frame in milliseconds
        parameter_filter: Optional parameter name to filter by
    """
    if not plots:
        print("No plots to create GIF from")
        return
    
    # Group plots by parameter
    param_plots = {}
    for (minibatch_idx, parameter_name), img in plots.items():
        if parameter_filter and parameter_name != parameter_filter:
            continue
        if parameter_name not in param_plots:
            param_plots[parameter_name] = []
        param_plots[parameter_name].append((minibatch_idx, img))
    
    # Sort by minibatch index for each parameter
    for param_name in param_plots:
        param_plots[param_name].sort(key=lambda x: x[0])
    
    # Create frames for GIF
    frames = []
    
    # Add frames for each parameter
    for param_name, param_data in param_plots.items():
        print(f"Adding {len(param_data)} frames for parameter: {param_name}")
        for minibatch_idx, img in param_data:
            frames.append(img)
    
    if frames:
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_path} with {len(frames)} frames")
    else:
        print("No frames to save")

def create_summary_analysis(csv_path, output_path=None):
    """
    Create a summary analysis showing slope trends across minibatches and parameters.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
        output_path: Path to save the summary plot
    """
    df = pd.read_csv(csv_path)
    
    # Calculate slopes for each parameter at each minibatch
    slopes_data = []
    
    grouped = df.groupby(['minibatch_idx', 'parameter_name'])
    
    for (minibatch_idx, parameter_name), group in grouped:
        subset_sizes = group['subset_size'].values
        x_values = 1/subset_sizes - 1/8
        y_values = group['residual_norm'].values
        
        if len(x_values) > 1:
            slope, _, r_value, _, _ = stats.linregress(x_values, y_values)
            slopes_data.append({
                'minibatch_idx': minibatch_idx,
                'parameter_name': parameter_name,
                'slope': slope,
                'r_squared': r_value**2
            })
    
    if not slopes_data:
        print("No slope data to analyze")
        return
    
    slopes_df = pd.DataFrame(slopes_data)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Slopes over time for each parameter
    ax1 = axes[0, 0]
    for param in slopes_df['parameter_name'].unique():
        param_data = slopes_df[slopes_df['parameter_name'] == param]
        ax1.plot(param_data['minibatch_idx'], param_data['slope'], 
                label=param, marker='o', markersize=3, alpha=0.7)
    ax1.set_xlabel('Minibatch Index')
    ax1.set_ylabel('Slope')
    ax1.set_title('Bias-Variance Slope Evolution')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: R-squared distribution
    ax2 = axes[0, 1]
    ax2.hist(slopes_df['r_squared'], bins=20, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('R-squared')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of R-squared Values')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Slope distribution
    ax3 = axes[1, 0]
    ax3.hist(slopes_df['slope'], bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Slope')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Slopes')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Slope vs R-squared
    ax4 = axes[1, 1]
    scatter = ax4.scatter(slopes_df['slope'], slopes_df['r_squared'], 
                         c=slopes_df['minibatch_idx'], cmap='viridis', alpha=0.6)
    ax4.set_xlabel('Slope')
    ax4.set_ylabel('R-squared')
    ax4.set_title('Slope vs R-squared (colored by minibatch)')
    plt.colorbar(scatter, ax=ax4, label='Minibatch Index')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Summary analysis saved to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean slope: {slopes_df['slope'].mean():.3e}")
    print(f"Std slope: {slopes_df['slope'].std():.3e}")
    print(f"Mean R-squared: {slopes_df['r_squared'].mean():.3f}")
    print(f"Parameters analyzed: {slopes_df['parameter_name'].nunique()}")
    print(f"Minibatches analyzed: {slopes_df['minibatch_idx'].nunique()}")

def main():
    parser = argparse.ArgumentParser(description='Analyze bias/variance tradeoff data')
    parser.add_argument('csv_file', help='Path to bias_variance_analysis.csv file')
    parser.add_argument('--output-dir', help='Directory to save individual plots')
    parser.add_argument('--gif-output', help='Path to save GIF animation')
    parser.add_argument('--parameter-filter', help='Filter GIF to specific parameter')
    parser.add_argument('--duration', type=int, default=500, help='GIF frame duration in ms')
    parser.add_argument('--summary', help='Path to save summary analysis plot')
    parser.add_argument('--summary-only', action='store_true', help='Only create summary analysis')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file {args.csv_file} not found")
        return
    
    if args.summary_only:
        create_summary_analysis(args.csv_file, args.summary)
        return
    
    # Process CSV and create plots
    plots = process_bias_variance_csv(args.csv_file, args.output_dir)
    
    # Create GIF if requested
    if args.gif_output:
        create_gif_from_plots(plots, args.gif_output, args.duration, args.parameter_filter)
    
    # Create summary analysis if requested
    if args.summary:
        create_summary_analysis(args.csv_file, args.summary)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()