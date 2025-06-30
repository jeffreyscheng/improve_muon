import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
import io
from scipy import stats
import seaborn as sns
import re
from collections import defaultdict

def parse_parameter_name(param_name):
    """
    Parse parameter name to extract layer index and parameter type.
    Examples: 
    - "blocks.0.mlp.fc_w" -> (0, "mlp.fc_w")
    - "blocks.0.attn.q" -> (0, "attn.q")
    - "blocks.0.attn.k" -> (0, "attn.k")
    """
    match = re.match(r'blocks\.(\d+)\.(.+)', param_name)
    if match:
        layer_idx = int(match.group(1))
        param_type = match.group(2)
        return layer_idx, param_type
    return None, param_name

def create_compound_layerwise_plot(minibatch_data, minibatch_idx):
    """
    Create a compound plot with 2x3 subplots showing bias-variance relationship for all parameter types.
    
    Args:
        minibatch_data: DataFrame with data for a specific minibatch, all parameter types
        minibatch_idx: The minibatch index
    
    Returns:
        PIL Image of the compound plot
    """
    # Define the expected parameter types and their subplot positions
    param_types = ['attn.q', 'attn.k', 'attn.v', 'attn.o', 'mlp.fc_w', 'mlp.proj_w']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get all layers for consistent coloring
    all_layers = sorted(minibatch_data['layer_idx'].unique()) if len(minibatch_data) > 0 else list(range(16))
    max_layer = max(all_layers) if all_layers else 15
    colors = plt.cm.viridis(np.array(all_layers) / max_layer)
    layer_color_map = {layer: colors[i] for i, layer in enumerate(all_layers)}
    
    for idx, param_type in enumerate(param_types):
        ax = axes[idx]
        param_data = minibatch_data[minibatch_data['param_type'] == param_type]
        
        if len(param_data) == 0:
            # Empty subplot
            ax.text(0.5, 0.5, f'No data for\n{param_type}', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{param_type}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            continue
        
        layers_in_data = sorted(param_data['layer_idx'].unique())
        
        for layer_idx in layers_in_data:
            layer_data = param_data[param_data['layer_idx'] == layer_idx]
            
            if len(layer_data) == 0:
                continue
                
            # Calculate (1/k - 1/8) for each subset size
            subset_sizes = layer_data['subset_size'].values
            x_values = 1/subset_sizes - 1/8
            y_values = layer_data['residual_norm'].values
            
            color = layer_color_map[layer_idx]
            
            # Scatter plot
            ax.scatter(x_values, y_values, alpha=0.6, s=20, color=color, 
                      label=f'L{layer_idx}' if idx == 0 else "")  # Only show legend on first subplot
            
        
        # Formatting for each subplot
        ax.set_xlabel('1/k - 1/8', fontsize=10)
        ax.set_ylabel('||Gradient Residual||_F', fontsize=10)
        ax.set_title(f'{param_type}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to the first subplot
        if idx == 0 and len(layers_in_data) > 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # Main title
    fig.suptitle(f'Bias-Variance Analysis - All Parameter Types\nMinibatch: {minibatch_idx}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def create_layerwise_bias_variance_plot(minibatch_data, minibatch_idx, param_type):
    """
    Create a plot showing bias-variance relationship for all layers of a specific parameter type.
    
    Args:
        minibatch_data: DataFrame with data for a specific minibatch and parameter type
        minibatch_idx: The minibatch index
        param_type: Parameter type (e.g., "mlp.fc_w")
    
    Returns:
        PIL Image of the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Get unique layers and create color map using viridis
    layers = sorted(minibatch_data['layer_idx'].unique())
    max_layer = max(layers) if layers else 15  # Assume 16 layers (0-15) if empty
    colors = plt.cm.viridis(np.array(layers) / max_layer)
    
    for i, layer_idx in enumerate(layers):
        layer_data = minibatch_data[minibatch_data['layer_idx'] == layer_idx]
        
        if len(layer_data) == 0:
            continue
            
        # Calculate (1/k - 1/8) for each subset size
        subset_sizes = layer_data['subset_size'].values
        x_values = 1/subset_sizes - 1/8
        y_values = layer_data['residual_norm'].values
        
        color = colors[i]
        
        # Scatter plot
        plt.scatter(x_values, y_values, alpha=0.6, s=30, color=color, 
                   label=f'Layer {layer_idx}')
        
    
    # Formatting
    plt.xlabel('1/k - 1/8', fontsize=12)
    plt.ylabel('||Gradient Residual||_F', fontsize=12)
    plt.title(f'Bias-Variance Analysis: {param_type}\nMinibatch: {minibatch_idx}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

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
    
    
    # Formatting
    plt.xlabel('1/k - 1/8', fontsize=12)
    plt.ylabel('||Gradient Residual||_F', fontsize=12)
    plt.title(f'Bias-Variance Analysis\nMinibatch: {minibatch_idx}, Parameter: {parameter_name}', 
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

def process_layerwise_bias_variance_csv(csv_path, save_individual=False, output_dir=None):
    """
    Process bias/variance CSV file and create layerwise plots for each parameter type at each minibatch.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
        save_individual: Whether to save individual PNG files (default False)
        output_dir: Directory to save plots (optional, only used if save_individual=True)
    
    Returns:
        Dictionary mapping (minibatch_idx, param_type) to PIL Images
    """
    df = pd.read_csv(csv_path)
    
    # Parse parameter names to extract layer index and parameter type
    parsed_data = []
    for _, row in df.iterrows():
        layer_idx, param_type = parse_parameter_name(row['parameter_name'])
        if layer_idx is not None:
            parsed_data.append({
                'minibatch_idx': row['minibatch_idx'],
                'parameter_name': row['parameter_name'],
                'layer_idx': layer_idx,
                'param_type': param_type,
                'subset_size': row['subset_size'],
                'subset_idx': row['subset_idx'],
                'residual_norm': row['residual_norm']
            })
    
    df_parsed = pd.DataFrame(parsed_data)
    
    if save_individual and output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    plots = {}
    
    # Group by minibatch and parameter type
    grouped = df_parsed.groupby(['minibatch_idx', 'param_type'])
    
    print(f"Processing {len(grouped)} parameter-type-minibatch combinations...")
    
    for (minibatch_idx, param_type), group in grouped:
        try:
            # Create layerwise plot
            img = create_layerwise_bias_variance_plot(group, minibatch_idx, param_type)
            plots[(minibatch_idx, param_type)] = img
            
            # Save individual plot only if requested
            if save_individual and output_dir:
                filename = f"minibatch_{minibatch_idx:06d}_{param_type.replace('.', '_')}.png"
                img.save(output_dir / filename)
                
        except Exception as e:
            print(f"Error processing minibatch {minibatch_idx}, parameter type {param_type}: {e}")
    
    return plots

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

def process_compound_layerwise_csv(csv_path):
    """
    Process bias/variance CSV file and create compound plots for all parameter types at each minibatch.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
    
    Returns:
        Dictionary mapping minibatch_idx to PIL Images
    """
    df = pd.read_csv(csv_path)
    
    # Parse parameter names to extract layer index and parameter type
    parsed_data = []
    for _, row in df.iterrows():
        layer_idx, param_type = parse_parameter_name(row['parameter_name'])
        if layer_idx is not None:
            parsed_data.append({
                'minibatch_idx': row['minibatch_idx'],
                'parameter_name': row['parameter_name'],
                'layer_idx': layer_idx,
                'param_type': param_type,
                'subset_size': row['subset_size'],
                'subset_idx': row['subset_idx'],
                'residual_norm': row['residual_norm']
            })
    
    df_parsed = pd.DataFrame(parsed_data)
    
    plots = {}
    
    # Group by minibatch only (all parameter types together)
    grouped = df_parsed.groupby('minibatch_idx')
    
    print(f"Processing {len(grouped)} minibatches for compound plots...")
    
    for minibatch_idx, group in grouped:
        try:
            # Create compound plot with all parameter types
            img = create_compound_layerwise_plot(group, minibatch_idx)
            plots[minibatch_idx] = img
                
        except Exception as e:
            print(f"Error processing minibatch {minibatch_idx}: {e}")
    
    return plots

def create_compound_layerwise_gif(csv_path, output_path="bias_variance_compound_evolution.gif", duration=200):
    """
    Create a single compound GIF with 2x3 subplots showing all parameter types.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
        output_path: Path to save the compound GIF
        duration: Duration per frame in milliseconds (default 200ms = 5fps)
    """
    # Process data to get compound plots
    plots = process_compound_layerwise_csv(csv_path)
    
    if not plots:
        print("No plots to create GIF from")
        return
    
    # Sort by minibatch index
    sorted_plots = sorted(plots.items())
    
    print(f"Creating compound GIF with {len(sorted_plots)} frames at {1000/duration:.1f} fps...")
    
    frames = [img for minibatch_idx, img in sorted_plots]
    
    if frames:
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"Compound GIF saved to {output_path}")

def create_layerwise_gifs(csv_path, output_dir=None, duration=200, save_individual_pngs=False):
    """
    Create separate GIF animations for each parameter type showing layerwise evolution.
    
    Args:
        csv_path: Path to the bias_variance_analysis.csv file
        output_dir: Directory to save GIFs (optional)
        duration: Duration per frame in milliseconds (default 200ms = 5fps)
        save_individual_pngs: Whether to save individual PNG frames
    """
    # Process data to get layerwise plots (don't save individual PNGs by default)
    plots = process_layerwise_bias_variance_csv(csv_path, save_individual_pngs, output_dir)
    
    if not plots:
        print("No plots to create GIFs from")
        return
    
    # Group plots by parameter type
    param_type_plots = defaultdict(list)
    for (minibatch_idx, param_type), img in plots.items():
        param_type_plots[param_type].append((minibatch_idx, img))
    
    # Sort by minibatch index for each parameter type
    for param_type in param_type_plots:
        param_type_plots[param_type].sort(key=lambda x: x[0])
    
    # Create separate GIF for each parameter type
    for param_type, param_data in param_type_plots.items():
        print(f"Creating GIF for {param_type} with {len(param_data)} frames at {1000/duration:.1f} fps...")
        
        frames = [img for minibatch_idx, img in param_data]
        
        if frames:
            # Create output path
            if output_dir:
                output_path = Path(output_dir) / f"{param_type.replace('.', '_')}_layerwise_evolution.gif"
            else:
                output_path = f"{param_type.replace('.', '_')}_layerwise_evolution.gif"
            
            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0
            )
            print(f"GIF saved to {output_path}")

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
    parser.add_argument('--duration', type=int, default=200, help='GIF frame duration in ms (default 200ms = 5fps)')
    parser.add_argument('--summary', help='Path to save summary analysis plot')
    parser.add_argument('--summary-only', action='store_true', help='Only create summary analysis')
    parser.add_argument('--layerwise-gifs', action='store_true', help='Create layerwise GIFs for each parameter type')
    parser.add_argument('--layerwise-only', action='store_true', help='Only create layerwise GIFs')
    parser.add_argument('--compound-gif', help='Create compound GIF with 2x3 subplot layout')
    parser.add_argument('--compound-only', action='store_true', help='Only create compound GIF')
    
    args = parser.parse_args()
    
    if not Path(args.csv_file).exists():
        print(f"Error: CSV file {args.csv_file} not found")
        return
    
    if args.summary_only:
        create_summary_analysis(args.csv_file, args.summary)
        return
    
    if args.compound_only or args.compound_gif:
        output_path = args.compound_gif if args.compound_gif else "bias_variance_compound_evolution.gif"
        create_compound_layerwise_gif(args.csv_file, output_path, args.duration)
        if args.compound_only:
            return
    
    if args.layerwise_only or args.layerwise_gifs:
        create_layerwise_gifs(args.csv_file, args.output_dir, args.duration)
        if args.layerwise_only:
            return
    
    # Process CSV and create plots
    if not args.layerwise_only and not args.compound_only:
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