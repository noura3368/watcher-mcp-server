import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Read the CSV data
#df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/multi_folder_commands_summary_all_folders.csv')
#df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/unique_commands_per_iteration.csv')
#df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/avg_unique_commands_per_folder.csv')
#df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/post_processing/plots/model_commands_oct14.csv')
df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt1_model_stats.csv')
df = df.dropna()

# Calculate correlation
correlation = df['size_billions'].corr(df['base_commands_seen'])

# Function to create a scatter plot
def create_scatter_plot(use_log_scale=False, suffix=""):
    # Create the scatter plot using plotly.express
    fig = px.scatter(
        df, 
        x='size_billions', 
        y='base_commands_seen',
        hover_data=['model'],
        title=f'Model Size vs Number of Valid Unique Commands{suffix}',
        labels={
            'size_billions': 'Model Size (Billions of Parameters)',
            'base_commands_seen': 'Number of Valid Unique Commands'
        },
        template='plotly_white',
        color_discrete_sequence=['#1f77b4'],
        opacity=0.7
    )
    
    # Update x-axis based on scale type
    if use_log_scale:
        fig.update_xaxes(
            type='log',
            tickvals=[0.001, 0.01, 0.1, 1, 2, 5, 10, 20, 32],
            ticktext=['0.001', '0.01', '0.1', '1', '2', '5', '10', '20', '32'],
            title='Model Size (Billions of Parameters)',
            showgrid=True
        )
    else:
        fig.update_xaxes(
            title='Model Size (Billions of Parameters)',
            showgrid=True
        )
    '''
    # Add trend line
    z = np.polyfit(df['size_billions'], df['avg_commands'], 1)
    x_trend = np.linspace(df['size_billions'].min(), df['size_billions'].max(), 100)
    y_trend = z[0] * x_trend + z[1]
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        #name=f'Trend Line (slope: {z[0]:.2f})',
        showlegend=True
    ))
    '''
    
    # Update layout
    fig.update_layout(
        width=1000,
        height=600,
        font=dict(family="Arial", size=12),
        title_font_size=16,
        hovermode='closest'
    )
    '''
    # Add correlation annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text=f'Correlation: {correlation:.3f}',
        showarrow=False,
        font=dict(size=12),
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='black',
        borderwidth=1

    )
    '''
    return fig

# Create both versions
print("Creating scatter plots...")

# 1. Linear scale version
fig_linear = create_scatter_plot(use_log_scale=False)
html_path_linear = '/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt1_scatter_plot.html'
fig_linear.write_html(html_path_linear)
print(f"Linear scale plot saved to: {html_path_linear}")

'''
# 2. Log scale version
fig_log = create_scatter_plot(use_log_scale=True, suffix=" (Log Scale)")
html_path_log = '/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/model_size_vs_avg_commands_log.html'
fig_log.write_html(html_path_log)
print(f"Log scale plot saved to: {html_path_log}")
'''
# Show both plots
