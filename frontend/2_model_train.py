"""
Model Training - Streamlit page for training ML models
Select models, set hyperparameters, and visualize training progress
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import requests
import json
import time
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Model Training - ML Playground",
    page_icon="🤖",
    layout="wide"
)

def check_data_availability() -> bool:
    """Check if data is available in session state"""
    if 'current_data' not in st.session_state:
        st.error("❌ No data loaded. Please go to Data Explorer to upload a dataset.")
        return False
    return True

def get_available_models() -> Dict[str, Dict[str, Any]]:
    """Get available ML models and their configurations"""
    return {
        "Linear Regression": {
            "type": "regression",
            "description": "Simple linear regression for continuous targets",
            "hyperparameters": {
                "fit_intercept": True,
                "normalize": False
            }
        },
        "Random Forest": {
            "type": "both",
            "description": "Ensemble method for both classification and regression",
            "hyperparameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 2,
                "min_samples_leaf": 1
            }
        },
        "Support Vector Machine": {
            "type": "both",
            "description": "Powerful algorithm for classification and regression",
            "hyperparameters": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            }
        },
        "Neural Network": {
            "type": "both",
            "description": "Multi-layer perceptron for complex patterns",
            "hyperparameters": {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "max_iter": 200
            }
        }
    }

def get_model_hyperparameters(model_name: str) -> Dict[str, Any]:
    """Get hyperparameters for a specific model"""
    models = get_available_models()
    return models.get(model_name, {}).get("hyperparameters", {})

def create_hyperparameter_ui(model_name: str) -> Dict[str, Any]:
    """Create UI for model hyperparameters"""
    st.subheader("⚙️ Model Hyperparameters")
    
    hyperparams = get_model_hyperparameters(model_name)
    user_params = {}
    
    if model_name == "Linear Regression":
        user_params["fit_intercept"] = st.checkbox("Fit Intercept", value=hyperparams["fit_intercept"])
        user_params["normalize"] = st.checkbox("Normalize", value=hyperparams["normalize"])
    
    elif model_name == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            user_params["n_estimators"] = st.slider("Number of Trees", 10, 500, hyperparams["n_estimators"])
            user_params["max_depth"] = st.slider("Max Depth", 1, 50, hyperparams["max_depth"])
        with col2:
            user_params["min_samples_split"] = st.slider("Min Samples Split", 2, 20, hyperparams["min_samples_split"])
            user_params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, hyperparams["min_samples_leaf"])
    
    elif model_name == "Support Vector Machine":
        col1, col2 = st.columns(2)
        with col1:
            user_params["C"] = st.slider("Regularization (C)", 0.1, 10.0, hyperparams["C"], step=0.1)
            user_params["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
        with col2:
            user_params["gamma"] = st.selectbox("Gamma", ["scale", "auto"], index=0)
    
    elif model_name == "Neural Network":
        col1, col2 = st.columns(2)
        with col1:
            hidden_layers = st.text_input("Hidden Layer Sizes (comma-separated)", "100,50")
            user_params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hidden_layers.split(","))
            user_params["activation"] = st.selectbox("Activation", ["relu", "tanh", "logistic"], index=0)
        with col2:
            user_params["solver"] = st.selectbox("Solver", ["adam", "sgd", "lbfgs"], index=0)
            user_params["alpha"] = st.slider("Alpha (L2 penalty)", 0.0001, 0.1, hyperparams["alpha"], step=0.0001)
            user_params["max_iter"] = st.slider("Max Iterations", 100, 1000, hyperparams["max_iter"], step=50)
    
    return user_params

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Prepare data for training and get target column"""
    st.subheader("🎯 Select Target Variable")
    
    # Get target column
    target_col = st.selectbox("Choose target variable:", df.columns.tolist())
    
    # Show target distribution
    if df[target_col].dtype in ['int64', 'float64']:
        fig = px.histogram(df, x=target_col, title=f"Distribution of {target_col}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(df[target_col].value_counts(), title=f"Distribution of {target_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    return df, target_col

def simulate_training_progress():
    """Simulate training progress with a progress bar"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(101):
        time.sleep(0.05)  # Simulate training time
        progress_bar.progress(i)
        if i < 30:
            status_text.text("Initializing model...")
        elif i < 60:
            status_text.text("Training model...")
        elif i < 90:
            status_text.text("Validating model...")
        else:
            status_text.text("Finalizing...")
    
    status_text.text("✅ Training completed!")
    return True

def create_training_curves():
    """Create mock training curves"""
    epochs = list(range(1, 101))
    
    # Simulate training and validation loss
    train_loss = [1.0 * np.exp(-epoch/30) + 0.1 * np.random.random() for epoch in epochs]
    val_loss = [1.2 * np.exp(-epoch/25) + 0.15 * np.random.random() for epoch in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', name='Training Loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', name='Validation Loss'))
    
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified'
    )
    
    return fig

def display_model_metrics():
    """Display model performance metrics"""
    st.subheader("📊 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "0.87", "0.02")
    with col2:
        st.metric("Precision", "0.85", "-0.01")
    with col3:
        st.metric("Recall", "0.89", "0.03")
    with col4:
        st.metric("F1-Score", "0.87", "0.01")

def main():
    """Main function for the Model Training page"""
    st.title("🤖 Model Training")
    st.markdown("Select a model, configure hyperparameters, and train on your dataset.")
    
    # Check if data is available
    if not check_data_availability():
        return
    
    df = st.session_state['current_data']
    
    # Model selection
    st.header("🎯 Select Your Model")
    models = get_available_models()
    
    model_name = st.selectbox(
        "Choose a model:",
        list(models.keys()),
        help="Select the machine learning model you want to train"
    )
    
    if model_name:
        model_info = models[model_name]
        st.info(f"**{model_name}**: {model_info['description']}")
        
        # Data preparation
        df, target_col = prepare_training_data(df)
        
        # Feature selection
        st.subheader("🔧 Feature Selection")
        feature_cols = st.multiselect(
            "Select features for training:",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col][:min(5, len(df.columns)-1)]
        )
        
        if not feature_cols:
            st.error("Please select at least one feature for training.")
            return
        
        # Hyperparameter tuning
        hyperparams = create_hyperparameter_ui(model_name)
        
        # Training configuration
        st.subheader("⚙️ Training Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 20)
            random_state = st.number_input("Random State", value=42, min_value=0)
        
        with col2:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
            early_stopping = st.checkbox("Early Stopping", value=True)
        
        # Start training
        st.subheader("🚀 Start Training")
        
        if st.button("🚀 Train Model", type="primary"):
            # Prepare training configuration
            training_config = {
                "model_name": model_name,
                "target_column": target_col,
                "feature_columns": feature_cols,
                "hyperparameters": hyperparams,
                "test_size": test_size / 100,
                "random_state": random_state,
                "cv_folds": cv_folds,
                "early_stopping": early_stopping
            }
            
            # Store in session state
            st.session_state['training_config'] = training_config
            
            # Simulate training
            with st.spinner("Training model..."):
                success = simulate_training_progress()
            
            if success:
                st.success("🎉 Model training completed successfully!")
                
                # Display results
                display_model_metrics()
                
                # Show training curves
                st.subheader("📈 Training Curves")
                fig = create_training_curves()
                st.plotly_chart(fig, use_container_width=True)
                
                # Model summary
                st.subheader("📋 Model Summary")
                st.json(training_config)
                
                # Save model option
                if st.button("💾 Save Model"):
                    st.info("Model saving feature coming soon!")
    
    else:
        st.info("👆 Please select a model to begin training")

if __name__ == "__main__":
    main() 