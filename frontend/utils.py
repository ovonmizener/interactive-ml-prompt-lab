"""
Shared UI components and helper functions for the Streamlit frontend
Common utilities used across all pages
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Union
import requests
import json
import time
import numpy as np
from datetime import datetime

def setup_page_config(page_title: str, page_icon: str, layout: str = "wide"):
    """Setup page configuration for Streamlit pages"""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )

def create_sidebar_navigation():
    """Create sidebar navigation for the application"""
    st.sidebar.title("🧪 ML Playground")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        [
            "📊 Data Explorer",
            "🤖 Model Training", 
            "💬 Prompt Workbench",
            "⚖️ Legal Search"
        ]
    )
    
    # Session state info
    st.sidebar.divider()
    st.sidebar.subheader("📈 Current Session")
    
    if 'current_data' in st.session_state:
        df = st.session_state['current_data']
        st.sidebar.metric("Dataset Rows", len(df))
        st.sidebar.metric("Dataset Columns", len(df.columns))
    else:
        st.sidebar.info("No data loaded")
    
    # Quick actions
    st.sidebar.divider()
    st.sidebar.subheader("⚡ Quick Actions")
    
    if st.sidebar.button("🔄 Clear Session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Session cleared!")
        st.rerun()
    
    if st.sidebar.button("📊 Load Sample Data"):
        # TODO: Load sample dataset
        st.sidebar.info("Sample data loading coming soon!")
    
    return page

def create_header(title: str, description: str):
    """Create a standardized header for pages"""
    st.title(title)
    st.markdown(description)
    st.divider()

def create_metric_card(title: str, value: Union[str, int, float], delta: Optional[str] = None):
    """Create a metric card with consistent styling"""
    st.metric(label=title, value=value, delta=delta)

def create_info_box(message: str, icon: str = "ℹ️"):
    """Create an info box with consistent styling"""
    st.info(f"{icon} {message}")

def create_success_box(message: str, icon: str = "✅"):
    """Create a success box with consistent styling"""
    st.success(f"{icon} {message}")

def create_warning_box(message: str, icon: str = "⚠️"):
    """Create a warning box with consistent styling"""
    st.warning(f"{icon} {message}")

def create_error_box(message: str, icon: str = "❌"):
    """Create an error box with consistent styling"""
    st.error(f"{icon} {message}")

def create_loading_spinner(message: str = "Processing..."):
    """Create a loading spinner with message"""
    return st.spinner(message)

def create_progress_bar():
    """Create a progress bar for long-running operations"""
    return st.progress(0)

def create_expandable_section(title: str, content_func, expanded: bool = False):
    """Create an expandable section with content"""
    with st.expander(title, expanded=expanded):
        content_func()

def create_tabs(tab_names: List[str]):
    """Create tabs with consistent styling"""
    return st.tabs(tab_names)

def create_columns(num_columns: int):
    """Create columns with consistent spacing"""
    return st.columns(num_columns)

def create_file_uploader(
    label: str,
    file_types: List[str],
    multiple: bool = False,
    help_text: str = ""
):
    """Create a file uploader with consistent styling"""
    return st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=multiple,
        help=help_text
    )

def create_selectbox(
    label: str,
    options: List[str],
    index: int = 0,
    help_text: str = ""
):
    """Create a selectbox with consistent styling"""
    return st.selectbox(
        label,
        options,
        index=index,
        help=help_text
    )

def create_multiselect(
    label: str,
    options: List[str],
    default: Optional[List[str]] = None,
    help_text: str = ""
):
    """Create a multiselect with consistent styling"""
    return st.multiselect(
        label,
        options,
        default=default,
        help=help_text
    )

def create_slider(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float = 1.0,
    help_text: str = ""
):
    """Create a slider with consistent styling"""
    return st.slider(
        label,
        min_value=min_value,
        max_value=max_value,
        value=value,
        step=step,
        help=help_text
    )

def create_text_input(
    label: str,
    value: str = "",
    placeholder: str = "",
    help_text: str = ""
):
    """Create a text input with consistent styling"""
    return st.text_input(
        label,
        value=value,
        placeholder=placeholder,
        help=help_text
    )

def create_text_area(
    label: str,
    value: str = "",
    height: int = 100,
    help_text: str = ""
):
    """Create a text area with consistent styling"""
    return st.text_area(
        label,
        value=value,
        height=height,
        help=help_text
    )

def create_button(
    label: str,
    type: str = "secondary",
    key: Optional[str] = None
):
    """Create a button with consistent styling"""
    return st.button(label, type=type, key=key)

def create_checkbox(
    label: str,
    value: bool = False,
    help_text: str = ""
):
    """Create a checkbox with consistent styling"""
    return st.checkbox(label, value=value, help=help_text)

def create_dataframe_display(
    df: pd.DataFrame,
    title: str = "Data Preview",
    max_rows: int = 10
):
    """Create a standardized dataframe display"""
    st.subheader(title)
    st.dataframe(df.head(max_rows), use_container_width=True)

def create_line_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Line Chart",
    color_column: Optional[str] = None
):
    """Create a line chart with consistent styling"""
    fig = px.line(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        title=title
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_bar_chart(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Bar Chart",
    color_column: Optional[str] = None
):
    """Create a bar chart with consistent styling"""
    fig = px.bar(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        title=title
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    st.plotly_chart(fig, use_container_width=True)

def create_histogram(
    data: pd.DataFrame,
    column: str,
    title: str = "Histogram",
    bins: int = 30
):
    """Create a histogram with consistent styling"""
    fig = px.histogram(
        data,
        x=column,
        title=title,
        nbins=bins
    )
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_scatter_plot(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str = "Scatter Plot",
    color_column: Optional[str] = None,
    size_column: Optional[str] = None
):
    """Create a scatter plot with consistent styling"""
    fig = px.scatter(
        data,
        x=x_column,
        y=y_column,
        color=color_column,
        size=size_column,
        title=title
    )
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title=y_column
    )
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(
    data: pd.DataFrame,
    title: str = "Correlation Matrix"
):
    """Create a correlation heatmap with consistent styling"""
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        title=title,
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def create_gauge_chart(
    value: float,
    title: str = "Gauge",
    min_value: float = 0,
    max_value: float = 1
):
    """Create a gauge chart with consistent styling"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={'axis': {'range': [min_value, max_value]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [min_value, max_value/2], 'color': "lightgray"},
                        {'range': [max_value/2, max_value*0.8], 'color': "yellow"},
                        {'range': [max_value*0.8, max_value], 'color': "green"}]}
    ))
    fig.update_layout(height=200)
    st.plotly_chart(fig, use_container_width=True)

def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """Format numbers with consistent styling"""
    if isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:.{decimals}f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentages with consistent styling"""
    return f"{value:.{decimals}f}%"

def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate dataframe and return validation results"""
    validation_results = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "info": {}
    }
    
    # Check for empty dataframe
    if df.empty:
        validation_results["is_valid"] = False
        validation_results["errors"].append("Dataframe is empty")
        return validation_results
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        validation_results["warnings"].append(f"Found {missing_counts.sum()} missing values")
        validation_results["info"]["missing_values"] = missing_counts.to_dict()
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        validation_results["warnings"].append(f"Found {duplicate_count} duplicate rows")
        validation_results["info"]["duplicate_count"] = duplicate_count
    
    # Check data types
    validation_results["info"]["data_types"] = df.dtypes.to_dict()
    validation_results["info"]["shape"] = df.shape
    
    return validation_results

def make_api_request(
    url: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30
) -> Optional[Dict[str, Any]]:
    """Make API request with error handling"""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON response: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def save_to_session_state(key: str, value: Any):
    """Save value to session state with validation"""
    st.session_state[key] = value

def get_from_session_state(key: str, default: Any = None) -> Any:
    """Get value from session state with default"""
    return st.session_state.get(key, default)

def clear_session_state():
    """Clear all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def create_download_button(
    data: Union[pd.DataFrame, str, bytes],
    filename: str,
    label: str = "Download",
    mime: str = "text/csv"
):
    """Create a download button with consistent styling"""
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
        mime = "text/csv"
    elif isinstance(data, str):
        mime = "text/plain"
    
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime
    ) 