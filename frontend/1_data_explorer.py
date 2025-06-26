"""
Data Explorer - Streamlit page for uploading and exploring datasets
Upload CSV/TXT files, show statistics, and create visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any
import requests
import io

# Page configuration
st.set_page_config(
    page_title="Data Explorer - ML Playground",
    page_icon="📊",
    layout="wide"
)

def load_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load data from uploaded file (CSV or TXT)
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        DataFrame if successful, None otherwise
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file, sep='\t')
        else:
            st.error("Unsupported file format. Please upload CSV or TXT files.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def display_data_overview(df: pd.DataFrame):
    """Display basic data overview and statistics"""
    st.subheader("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

def display_data_types(df: pd.DataFrame):
    """Display data types and basic info for each column"""
    st.subheader("🔍 Column Information")
    
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    
    st.dataframe(info_df, use_container_width=True)

def create_histograms(df: pd.DataFrame):
    """Create histograms for numerical columns"""
    st.subheader("📊 Data Distributions")
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if len(numerical_cols) == 0:
        st.info("No numerical columns found for histogram visualization.")
        return
    
    # Create histograms for each numerical column
    for col in numerical_cols:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True)

def create_correlation_matrix(df: pd.DataFrame):
    """Create correlation matrix heatmap for numerical columns"""
    st.subheader("🔗 Correlation Matrix")
    
    numerical_df = df.select_dtypes(include=['number'])
    
    if len(numerical_df.columns) < 2:
        st.info("Need at least 2 numerical columns for correlation analysis.")
        return
    
    corr_matrix = numerical_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function for the Data Explorer page"""
    st.title("📊 Data Explorer")
    st.markdown("Upload your dataset and explore its characteristics with interactive visualizations.")
    
    # File upload section
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or TXT file",
        type=['csv', 'txt'],
        help="Upload your dataset to begin exploration"
    )
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            st.success(f"✅ Successfully loaded {uploaded_file.name}")
            
            # Store data in session state for other pages
            st.session_state['current_data'] = df
            st.session_state['data_filename'] = uploaded_file.name
            
            # Display data overview
            display_data_overview(df)
            
            # Show first few rows
            st.subheader("👀 First Few Rows")
            st.dataframe(df.head(), use_container_width=True)
            
            # Column information
            display_data_types(df)
            
            # Visualizations
            create_histograms(df)
            create_correlation_matrix(df)
            
            # Data quality insights
            st.subheader("🔍 Data Quality Insights")
            
            # Missing values analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.warning("⚠️ Missing values detected:")
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': (missing_data.values / len(df)) * 100
                }).sort_values('Missing Count', ascending=False)
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                st.warning(f"⚠️ {duplicate_count} duplicate rows found")
            else:
                st.success("✅ No duplicate rows found!")
    
    else:
        st.info("👆 Please upload a file to begin data exploration")
        
        # Show sample data option
        if st.button("Load Sample Dataset"):
            # TODO: Load sample dataset for demonstration
            st.info("Sample dataset feature coming soon!")

if __name__ == "__main__":
    main() 