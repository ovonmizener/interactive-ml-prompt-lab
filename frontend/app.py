"""
Interactive ML & Prompting Playground - Main App
Unified Streamlit application with multipage navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure the page
st.set_page_config(
    page_title="Interactive ML & Prompting Playground",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for shared data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = None
if 'llm_responses' not in st.session_state:
    st.session_state.llm_responses = []
if 'search_results' not in st.session_state:
    st.session_state.search_results = []

def main():
    """Main application entry point"""
    
    # Sidebar navigation
    st.sidebar.title("🤖 ML & Prompting Playground")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "💬 Prompt Workbench", "⚖️ Legal Search"]
    )
    
    # Display current page
    if page == "🏠 Home":
        show_home()
    elif page == "📊 Data Explorer":
        show_data_explorer()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "💬 Prompt Workbench":
        show_prompt_workbench()
    elif page == "⚖️ Legal Search":
        show_legal_search()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Current Status")
    
    # Show current data status
    if st.session_state.uploaded_data is not None:
        st.sidebar.success(f"✅ Data loaded: {len(st.session_state.uploaded_data)} rows")
    else:
        st.sidebar.warning("⚠️ No data uploaded")
    
    # Show model status
    if st.session_state.trained_model is not None:
        st.sidebar.success("✅ Model trained")
    else:
        st.sidebar.info("ℹ️ No model trained yet")

def show_home():
    """Home page with overview and quick actions"""
    st.title("🏠 Interactive ML & Prompting Playground")
    st.markdown("Welcome to your comprehensive machine learning and prompt engineering workspace!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **📊 Data Explorer**: Upload and explore your dataset
        2. **🤖 Model Training**: Train ML models on your data
        3. **💬 Prompt Workbench**: Experiment with LLM prompts
        4. **⚖️ Legal Search**: Search through legal documents
        """)
        
        st.info("👆 Use the sidebar to navigate between features")
    
    with col2:
        st.markdown("### 📈 Current Status")
        
        # Data status
        if st.session_state.uploaded_data is not None:
            st.success(f"✅ Data loaded: {len(st.session_state.uploaded_data)} rows")
            st.dataframe(st.session_state.uploaded_data.head())
        else:
            st.info("ℹ️ No data uploaded yet")
        
        # Model status
        if st.session_state.trained_model is not None:
            st.success("✅ Model trained and ready")
        else:
            st.info("ℹ️ No model trained yet")
    
    st.markdown("---")
    st.markdown("### 🎯 Features Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**📊 Data Explorer**")
        st.markdown("- Upload CSV/Excel files")
        st.markdown("- Data visualization")
        st.markdown("- Statistical analysis")
    
    with col2:
        st.markdown("**🤖 Model Training**")
        st.markdown("- Multiple ML algorithms")
        st.markdown("- Hyperparameter tuning")
        st.markdown("- Model evaluation")
    
    with col3:
        st.markdown("**💬 Prompt Workbench**")
        st.markdown("- LLM integration")
        st.markdown("- Prompt engineering")
        st.markdown("- Response analysis")
    
    with col4:
        st.markdown("**⚖️ Legal Search**")
        st.markdown("- Document search")
        st.markdown("- Semantic similarity")
        st.markdown("- Legal analysis")

def show_data_explorer():
    """Data Explorer page"""
    st.title("📊 Data Explorer")
    st.markdown("Upload, explore, and analyze your datasets")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset to get started"
    )
    
    if uploaded_file is not None:
        try:
            # Load data based on file type
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Store in session state
            st.session_state.uploaded_data = data
            
            st.success(f"✅ Successfully loaded {len(data)} rows and {len(data.columns)} columns")
            
            # Data overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(data))
            with col2:
                st.metric("Columns", len(data.columns))
            with col3:
                st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(data.head(10))
            
            # Data info
            st.subheader("📊 Data Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Data Types:**")
                st.write(data.dtypes)
            
            with col2:
                st.write("**Missing Values:**")
                missing_data = data.isnull().sum()
                st.write(missing_data[missing_data > 0])
            
            # Basic statistics
            st.subheader("📈 Basic Statistics")
            st.write(data.describe())
            
            # Data visualization
            st.subheader("📊 Data Visualization")
            
            # Select column for histogram
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for histogram:", numeric_cols)
                if selected_col:
                    st.histogram_chart(data[selected_col])
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    else:
        st.info("👆 Please upload a dataset to get started")

def show_model_training():
    """Model Training page"""
    st.title("🤖 Model Training")
    st.markdown("Train and evaluate machine learning models")
    
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data in the Data Explorer first!")
        return
    
    data = st.session_state.uploaded_data
    
    st.success(f"✅ Using dataset with {len(data)} rows and {len(data.columns)} columns")
    
    # Data type analysis
    st.subheader("📊 Data Type Analysis")
    
    # Get numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**✅ Numeric Columns (Available for features):**")
        if numeric_cols:
            for col in numeric_cols:
                st.write(f"• {col} ({data[col].dtype})")
        else:
            st.warning("No numeric columns found!")
    
    with col2:
        st.write("**⚠️ Non-Numeric Columns (Not available for features):**")
        if non_numeric_cols:
            for col in non_numeric_cols:
                st.write(f"• {col} ({data[col].dtype})")
        else:
            st.info("All columns are numeric!")
    
    # Data quality check
    st.subheader("🔍 Data Quality Check")
    
    # Check for missing values
    missing_data = data.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    
    if len(missing_cols) > 0:
        st.warning("⚠️ Missing values detected:")
        for col, count in missing_cols.items():
            percentage = (count / len(data)) * 100
            st.write(f"• {col}: {count} missing values ({percentage:.1f}%)")
        
        # Show missing value heatmap
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis', ax=ax)
        ax.set_title('Missing Values Heatmap')
        st.pyplot(fig)
        plt.close()
    else:
        st.success("✅ No missing values found in the dataset!")
    
    # Check for infinite values in numeric columns
    if numeric_cols:
        inf_cols = []
        for col in numeric_cols:
            if np.isinf(data[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            st.warning("⚠️ Infinite values detected in numeric columns:")
            for col in inf_cols:
                st.write(f"• {col}")
        else:
            st.success("✅ No infinite values found in numeric columns!")
    
    # Model configuration
    st.subheader("⚙️ Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column (can be any column)
        target_col = st.selectbox("Select target column:", data.columns)
        
        # Select features (only numeric columns)
        if numeric_cols:
            # Remove target column from feature options if it's numeric
            available_features = [col for col in numeric_cols if col != target_col]
            
            if available_features:
                feature_cols = st.multiselect(
                    "Select feature columns (numeric only):",
                    available_features,
                    default=available_features[:min(3, len(available_features))] if available_features else []
                )
            else:
                st.warning("⚠️ No numeric columns available for features (target column is the only numeric column)")
                feature_cols = []
        else:
            st.error("❌ No numeric columns found in the dataset! Cannot train ML models.")
            return
    
    with col2:
        # Model type
        model_type = st.selectbox(
            "Select model type:",
            ["Classification", "Regression"]
        )
        
        # Algorithm
        if model_type == "Classification":
            algorithm = st.selectbox(
                "Select algorithm:",
                ["Random Forest", "Logistic Regression", "SVM", "Decision Tree"]
            )
        else:
            algorithm = st.selectbox(
                "Select algorithm:",
                ["Random Forest", "Linear Regression", "SVR", "Decision Tree"]
            )
    
    # Training parameters
    st.subheader("🎯 Training Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        test_size = st.slider("Test size:", 0.1, 0.5, 0.2, 0.1)
    
    with col2:
        random_state = st.number_input("Random state:", value=42, min_value=0)
    
    with col3:
        if algorithm == "Random Forest":
            n_estimators = st.number_input("Number of estimators:", value=100, min_value=1)
    
    # Train model
    if st.button("🚀 Train Model", type="primary"):
        if not feature_cols:
            st.error("Please select at least one feature column!")
            return
        
        with st.spinner("Training model..."):
            try:
                # Import required libraries
                from sklearn.model_selection import train_test_split
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.linear_model import LogisticRegression, LinearRegression
                from sklearn.svm import SVC, SVR
                from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
                from sklearn.preprocessing import LabelEncoder
                
                # Prepare data with better error handling
                X = data[feature_cols].copy()
                y = data[target_col].copy()
                
                # Remove rows with missing values in either features or target
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(X) == 0:
                    st.error("❌ No valid data remaining after removing rows with missing values!")
                    return
                
                st.info(f"📊 Using {len(X)} rows after removing {len(data) - len(X)} rows with missing values.")
                
                # Handle missing values in features (should be minimal now, but just in case)
                X = X.fillna(X.mean())
                
                # Handle target variable if it's categorical
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                    st.info(f"Target column '{target_col}' was categorical and has been encoded.")
                
                # Final validation
                if X.isnull().any().any() or y.isnull().any():
                    st.error("❌ Data still contains missing values after cleaning. Please check your dataset.")
                    return
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if model_type == "Classification" else None
                )
                
                # Select and train model
                if model_type == "Classification":
                    if algorithm == "Random Forest":
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                    elif algorithm == "Logistic Regression":
                        model = LogisticRegression(random_state=random_state, max_iter=1000)
                    elif algorithm == "SVM":
                        model = SVC(random_state=random_state)
                    elif algorithm == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=random_state)
                else:
                    if algorithm == "Random Forest":
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
                    elif algorithm == "Linear Regression":
                        model = LinearRegression()
                    elif algorithm == "SVR":
                        model = SVR()
                    elif algorithm == "Decision Tree":
                        model = DecisionTreeRegressor(random_state=random_state)
                
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Store model and results
                st.session_state.trained_model = {
                    'model': model,
                    'algorithm': algorithm,
                    'model_type': model_type,
                    'feature_cols': feature_cols,
                    'target_col': target_col,
                    'X_test': X_test,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                st.success("✅ Model trained successfully!")
                
                # Display results
                st.subheader("📊 Model Performance")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if model_type == "Classification":
                        accuracy = accuracy_score(y_test, y_pred)
                        st.metric("Accuracy", f"{accuracy:.3f}")
                        st.write("Classification Report:")
                        st.text(classification_report(y_test, y_pred))
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = np.sqrt(mse)
                        st.metric("Mean Squared Error", f"{mse:.3f}")
                        st.metric("Root Mean Squared Error", f"{rmse:.3f}")
                
                with col2:
                    # Feature importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': feature_cols,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        st.write("Feature Importance:")
                        st.bar_chart(importance_df.set_index('Feature'))
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                st.info("💡 Tip: Make sure your target column is appropriate for the selected model type (classification vs regression).")
    
    # Model predictions
    if st.session_state.trained_model is not None:
        st.subheader("🔮 Make Predictions")
        
        model_info = st.session_state.trained_model
        
        # Create input form for new predictions
        st.write("Enter values for prediction:")
        
        input_data = {}
        cols = st.columns(len(model_info['feature_cols']))
        
        for i, col in enumerate(model_info['feature_cols']):
            with cols[i]:
                # Get the range of values for better input
                col_data = data[col]
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())
                
                input_data[col] = st.number_input(
                    f"{col}:", 
                    value=mean_val,
                    min_value=min_val,
                    max_value=max_val,
                    step=(max_val - min_val) / 100
                )
        
        if st.button("🔮 Predict"):
            try:
                # Create input array
                input_array = np.array([[input_data[col] for col in model_info['feature_cols']]])
                
                # Make prediction
                prediction = model_info['model'].predict(input_array)[0]
                
                st.success(f"Prediction: {prediction}")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def show_prompt_workbench():
    """Prompt Workbench page"""
    st.title("💬 Prompt Workbench")
    st.markdown("Experiment with LLM prompts and responses")
    
    # Prompt input
    st.subheader("✍️ Prompt Input")
    
    prompt = st.text_area(
        "Enter your prompt:",
        placeholder="Enter your prompt here...",
        height=150
    )
    
    # Model selection
    col1, col2 = st.columns(2)
    
    with col1:
        model = st.selectbox(
            "Select LLM model:",
            ["gpt-3.5-turbo", "gpt-4", "claude-3", "llama-2"]
        )
    
    with col2:
        temperature = st.slider("Temperature:", 0.0, 2.0, 0.7, 0.1)
    
    # Generate response
    if st.button("🚀 Generate Response", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt!")
            return
        
        with st.spinner("Generating response..."):
            try:
                # Simulate LLM response (replace with actual API call)
                import time
                time.sleep(1)  # Simulate API delay
                
                # Mock response based on prompt
                if "hello" in prompt.lower():
                    response = "Hello! How can I help you today?"
                elif "explain" in prompt.lower():
                    response = "I'd be happy to explain that concept. Could you provide more specific details about what you'd like me to explain?"
                elif "code" in prompt.lower():
                    response = "Here's a sample code snippet:\n\n```python\nprint('Hello, World!')\n```"
                else:
                    response = "This is a simulated response. In a real implementation, this would be the actual LLM response to your prompt."
                
                # Store response
                st.session_state.llm_responses.append({
                    'prompt': prompt,
                    'response': response,
                    'model': model,
                    'temperature': temperature
                })
                
                st.success("✅ Response generated!")
                
                # Display response
                st.subheader("🤖 Generated Response")
                st.write(response)
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    # Response history
    if st.session_state.llm_responses:
        st.subheader("📚 Response History")
        
        for i, resp in enumerate(reversed(st.session_state.llm_responses)):
            with st.expander(f"Response {len(st.session_state.llm_responses) - i} - {resp['model']}"):
                st.write("**Prompt:**")
                st.write(resp['prompt'])
                st.write("**Response:**")
                st.write(resp['response'])
                st.write(f"**Model:** {resp['model']} | **Temperature:** {resp['temperature']}")

def show_legal_search():
    """Legal Search page"""
    st.title("⚖️ Legal Search")
    st.markdown("Search and analyze legal documents")
    
    # Search interface
    st.subheader("🔍 Document Search")
    
    # Sample legal documents (in real app, these would be loaded from a database)
    sample_docs = [
        "The defendant is hereby ordered to pay damages in the amount of $50,000.",
        "This contract is subject to the laws of the State of California.",
        "The plaintiff alleges breach of contract and seeks specific performance.",
        "All disputes shall be resolved through binding arbitration.",
        "The parties agree to maintain confidentiality of all proprietary information."
    ]
    
    # Search query
    query = st.text_input("Enter search query:", placeholder="e.g., breach of contract")
    
    # Search options
    col1, col2 = st.columns(2)
    
    with col1:
        search_type = st.selectbox(
            "Search type:",
            ["Semantic Search", "Keyword Search", "Exact Match"]
        )
    
    with col2:
        max_results = st.slider("Max results:", 1, 10, 5)
    
    # Perform search
    if st.button("🔍 Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query!")
            return
        
        with st.spinner("Searching documents..."):
            try:
                # Simple search simulation
                import re
                
                results = []
                for i, doc in enumerate(sample_docs):
                    score = 0
                    
                    if search_type == "Keyword Search":
                        # Simple keyword matching
                        query_words = query.lower().split()
                        doc_words = doc.lower().split()
                        matches = sum(1 for word in query_words if word in doc_words)
                        score = matches / len(query_words) if query_words else 0
                    
                    elif search_type == "Exact Match":
                        # Exact phrase matching
                        if query.lower() in doc.lower():
                            score = 1.0
                    
                    else:  # Semantic Search
                        # Simple semantic similarity (word overlap)
                        query_words = set(re.findall(r'\w+', query.lower()))
                        doc_words = set(re.findall(r'\w+', doc.lower()))
                        intersection = query_words.intersection(doc_words)
                        union = query_words.union(doc_words)
                        score = len(intersection) / len(union) if union else 0
                    
                    if score > 0:
                        results.append({
                            'document': doc,
                            'score': score,
                            'id': i
                        })
                
                # Sort by score and limit results
                results.sort(key=lambda x: x['score'], reverse=True)
                results = results[:max_results]
                
                # Store results
                st.session_state.search_results = results
                
                st.success(f"✅ Found {len(results)} relevant documents!")
                
                # Display results
                st.subheader("📄 Search Results")
                
                for i, result in enumerate(results):
                    with st.expander(f"Document {result['id'] + 1} (Score: {result['score']:.3f})"):
                        st.write(result['document'])
                        
                        # Highlight matching terms
                        if search_type == "Keyword Search":
                            highlighted_doc = result['document']
                            for word in query.lower().split():
                                highlighted_doc = re.sub(
                                    f'({word})',
                                    r'**\1**',
                                    highlighted_doc,
                                    flags=re.IGNORECASE
                                )
                            st.markdown(highlighted_doc)
                
            except Exception as e:
                st.error(f"Error performing search: {str(e)}")
    
    # Search history
    if st.session_state.search_results:
        st.subheader("📚 Search History")
        st.write(f"Last search returned {len(st.session_state.search_results)} results")

if __name__ == "__main__":
    main() 