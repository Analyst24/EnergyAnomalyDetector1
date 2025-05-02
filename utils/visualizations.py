import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import base64

def create_energy_overview(df, y_column='consumption', date_column='timestamp'):
    """
    Create an energy consumption overview chart using Plotly
    
    Args:
        df: pandas DataFrame with energy data
        y_column: column name for energy consumption
        date_column: column name for timestamp
        
    Returns:
        Plotly figure object
    """
    if date_column not in df.columns or y_column not in df.columns:
        return None
    
    # Make a copy to avoid modifying the original
    df_plot = df.copy()
    
    # Ensure timestamp is datetime
    if date_column in df_plot.columns:
        df_plot[date_column] = pd.to_datetime(df_plot[date_column])
    
    # Create figure
    fig = px.line(df_plot, x=date_column, y=y_column, 
                 title='Energy Consumption Overview',
                 labels={y_column: 'Energy Consumption', date_column: 'Time'},
                 template='plotly_dark')
    
    # Add anomalies if present
    if 'anomaly' in df_plot.columns:
        anomaly_points = df_plot[df_plot['anomaly'] == 1]
        
        fig.add_scatter(x=anomaly_points[date_column], 
                       y=anomaly_points[y_column],
                       mode='markers',
                       marker=dict(color='red', size=10, symbol='circle-open'),
                       name='Anomalies')
    
    # Update layout
    fig.update_layout(
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode='closest',
        xaxis_title='Time',
        yaxis_title='Energy Consumption'
    )
    
    return fig

def create_anomaly_distribution(df):
    """
    Create an anomaly distribution chart
    
    Args:
        df: pandas DataFrame with anomaly data
        
    Returns:
        Plotly figure object
    """
    if 'anomaly' not in df.columns:
        return None
    
    # Count anomalies vs normal points
    anomaly_counts = df['anomaly'].value_counts().reset_index()
    anomaly_counts.columns = ['Type', 'Count']
    anomaly_counts['Type'] = anomaly_counts['Type'].map({0: 'Normal', 1: 'Anomaly'})
    
    # Create figure
    fig = px.pie(anomaly_counts, values='Count', names='Type', 
                title='Anomaly Distribution',
                color='Type',
                color_discrete_map={'Normal': '#2ECC71', 'Anomaly': '#E74C3C'},
                template='plotly_dark')
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', y=1.1)
    )
    
    return fig

def create_hourly_analysis(df):
    """
    Create a time of day analysis chart
    
    Args:
        df: pandas DataFrame with timestamp and anomaly data
        
    Returns:
        Plotly figure object
    """
    if 'timestamp' not in df.columns or 'anomaly' not in df.columns:
        return None
    
    # Make a copy to avoid modifying the original
    df_plot = df.copy()
    
    # Ensure timestamp is datetime
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    df_plot['hour'] = df_plot['timestamp'].dt.hour
    
    # Group by hour and count anomalies
    hourly_data = df_plot.groupby('hour')['anomaly'].agg(['sum', 'count']).reset_index()
    hourly_data['normal'] = hourly_data['count'] - hourly_data['sum']
    hourly_data['anomaly_rate'] = hourly_data['sum'] / hourly_data['count'] * 100
    
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart for counts
    fig.add_trace(
        go.Bar(x=hourly_data['hour'], y=hourly_data['normal'], name='Normal',
              marker_color='#2ECC71'),
        secondary_y=False
    )
    fig.add_trace(
        go.Bar(x=hourly_data['hour'], y=hourly_data['sum'], name='Anomalies',
              marker_color='#E74C3C'),
        secondary_y=False
    )
    
    # Add line chart for anomaly rate
    fig.add_trace(
        go.Scatter(x=hourly_data['hour'], y=hourly_data['anomaly_rate'], name='Anomaly Rate (%)',
                  line=dict(color='#F39C12', width=2, dash='dot')),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Time of Day Analysis',
        template='plotly_dark',
        barmode='stack',
        xaxis=dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=1),
        yaxis=dict(title='Number of Data Points'),
        yaxis2=dict(title='Anomaly Rate (%)', range=[0, max(hourly_data['anomaly_rate']) * 1.2]),
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_feature_correlation(df):
    """
    Create a feature correlation heatmap
    
    Args:
        df: pandas DataFrame with numeric features
        
    Returns:
        Plotly figure object
    """
    # Select numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Remove anomaly and anomaly_score if present (for a cleaner correlation matrix)
    cols_to_drop = ['anomaly', 'anomaly_score']
    numeric_df = numeric_df.drop([col for col in cols_to_drop if col in numeric_df.columns], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = px.imshow(corr_matrix,
                   text_auto='.2f',
                   aspect='auto',
                   color_continuous_scale='RdBu_r',
                   title='Feature Correlation Matrix',
                   template='plotly_dark')
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )
    
    return fig

def create_confusion_matrix(true_labels, predictions, title='Confusion Matrix'):
    """
    Create a confusion matrix visualization
    
    Args:
        true_labels: array of true labels
        predictions: array of predicted labels
        title: title for the plot
        
    Returns:
        Plotly figure object
    """
    # Calculate confusion matrix
    z = [[0, 0], [0, 0]]
    
    for t, p in zip(true_labels, predictions):
        z[t][p] += 1
    
    x = ['Normal', 'Anomaly']
    y = ['Normal', 'Anomaly']
    
    # Create heatmap
    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z, colorscale='Blues'
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis=dict(title='Predicted'),
        yaxis=dict(title='Actual'),
        template='plotly_dark',
        margin=dict(l=40, r=20, t=60, b=20)
    )
    
    return fig

def create_feature_importance(feature_names, importance_values):
    """
    Create feature importance bar chart
    
    Args:
        feature_names: list of feature names
        importance_values: corresponding importance values
        
    Returns:
        Plotly figure object
    """
    # Sort features by importance
    sorted_idx = np.argsort(importance_values)
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importance_values[i] for i in sorted_idx]
    
    # Create bar chart
    fig = px.bar(
        x=sorted_importance,
        y=sorted_features,
        orientation='h',
        title='Feature Importance',
        labels={'x': 'Importance', 'y': 'Feature'},
        template='plotly_dark'
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=max(300, len(feature_names) * 25)  # Dynamic height based on feature count
    )
    
    return fig

def create_anomaly_scores_histogram(df):
    """
    Create histogram of anomaly scores
    
    Args:
        df: pandas DataFrame with anomaly_score column
        
    Returns:
        Plotly figure object
    """
    if 'anomaly_score' not in df.columns:
        return None
    
    # Create histogram
    fig = px.histogram(
        df, x='anomaly_score', 
        color='anomaly' if 'anomaly' in df.columns else None,
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        marginal='box',
        title='Distribution of Anomaly Scores',
        template='plotly_dark'
    )
    
    # Add threshold line if available
    if 'threshold' in df.columns:
        threshold = df['threshold'].iloc[0]
        fig.add_vline(x=threshold, line_dash='dash', line_color='white',
                     annotation_text=f'Threshold: {threshold:.2f}')
    
    # Update layout
    fig.update_layout(
        xaxis_title='Anomaly Score',
        yaxis_title='Count',
        legend_title='Anomaly',
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_3d_cluster_plot(df, x_col, y_col, z_col):
    """
    Create 3D scatter plot for anomaly visualization
    
    Args:
        df: pandas DataFrame with data points
        x_col, y_col, z_col: column names for the 3 dimensions
        
    Returns:
        Plotly figure object
    """
    if not all(col in df.columns for col in [x_col, y_col, z_col, 'anomaly']):
        return None
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col,
        color='anomaly',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        title='3D Anomaly Visualization',
        labels={x_col: x_col, y_col: y_col, z_col: z_col},
        template='plotly_dark'
    )
    
    # Update marker size and opacity
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    
    # Update layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        )
    )
    
    return fig

def create_animated_anomaly_plot(df, date_column='timestamp', y_column='consumption'):
    """
    Create an animated scatter plot showing anomaly detection over time
    
    Args:
        df: pandas DataFrame with time series data and anomalies
        date_column: column name for timestamp
        y_column: column name for metric to plot
        
    Returns:
        Plotly figure object
    """
    if not all(col in df.columns for col in [date_column, y_column, 'anomaly']):
        return None
    
    # Ensure timestamp is datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Create time-based features for animation frames
    df['date'] = df[date_column].dt.date
    
    # Create animated scatter plot
    fig = px.scatter(
        df, x=date_column, y=y_column,
        color='anomaly',
        color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
        animation_frame='date',
        title=f'Anomaly Detection Over Time: {y_column}',
        template='plotly_dark',
        labels={date_column: 'Time', y_column: y_column, 'anomaly': 'Anomaly'}
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title=y_column,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Configure animation
    fig.update_layout(
        updatemenus=[
            dict(
                type='buttons',
                showactive=False,
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True}]
                    ),
                    dict(
                        label='Pause',
                        method='animate',
                        args=[[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
                    )
                ]
            )
        ]
    )
    
    return fig
