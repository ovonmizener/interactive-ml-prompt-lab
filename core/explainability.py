"""
Explainability - Model interpretability and explainability tools
SHAP analysis, attention maps, and feature importance visualization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging
import warnings

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) analysis utilities"""
    
    def __init__(self):
        self.explainer = None
        self.shap_values = None
        self.feature_names = None
    
    def create_explainer(
        self,
        model,
        X_train: pd.DataFrame,
        model_type: str = "tree"
    ) -> shap.Explainer:
        """
        Create SHAP explainer for the model
        
        Args:
            model: Trained machine learning model
            X_train: Training features
            model_type: Type of model ('tree', 'linear', 'kernel')
            
        Returns:
            SHAP explainer object
        """
        try:
            if model_type == "tree":
                self.explainer = shap.TreeExplainer(model)
            elif model_type == "linear":
                self.explainer = shap.LinearExplainer(model, X_train)
            elif model_type == "kernel":
                self.explainer = shap.KernelExplainer(model.predict, X_train)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            self.feature_names = X_train.columns.tolist()
            logger.info(f"SHAP explainer created for {model_type} model")
            return self.explainer
        
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {str(e)}")
            raise
    
    def calculate_shap_values(
        self,
        X: pd.DataFrame,
        sample_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Calculate SHAP values for given data
        
        Args:
            X: Input features
            sample_size: Number of samples to use (for large datasets)
            
        Returns:
            SHAP values array
        """
        try:
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized. Call create_explainer first.")
            
            if sample_size and len(X) > sample_size:
                X_sample = X.sample(n=sample_size, random_state=42)
            else:
                X_sample = X
            
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[0]  # Take first class for classification
            
            logger.info(f"SHAP values calculated for {len(X_sample)} samples")
            return self.shap_values
        
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")
            raise
    
    def plot_summary(
        self,
        X: pd.DataFrame,
        max_display: int = 20,
        plot_type: str = "bar"
    ) -> go.Figure:
        """
        Create SHAP summary plot
        
        Args:
            X: Input features
            max_display: Maximum number of features to display
            plot_type: Type of plot ('bar', 'dot', 'violin')
            
        Returns:
            Plotly figure object
        """
        try:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(self.shap_values).mean(0)
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=True)
            
            # Limit to max_display features
            if len(feature_importance) > max_display:
                feature_importance = feature_importance.tail(max_display)
            
            # Create plot
            fig = go.Figure()
            
            if plot_type == "bar":
                fig.add_trace(go.Bar(
                    x=feature_importance['Importance'],
                    y=feature_importance['Feature'],
                    orientation='h',
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title="SHAP Feature Importance",
                    xaxis_title="Mean |SHAP Value|",
                    yaxis_title="Features",
                    height=400 + len(feature_importance) * 20
                )
            
            elif plot_type == "dot":
                # Create scatter plot of SHAP values
                for i, feature in enumerate(feature_importance['Feature']):
                    feature_idx = self.feature_names.index(feature)
                    fig.add_trace(go.Scatter(
                        x=X.iloc[:, feature_idx],
                        y=self.shap_values[:, feature_idx],
                        mode='markers',
                        name=feature,
                        opacity=0.6
                    ))
                
                fig.update_layout(
                    title="SHAP Values vs Feature Values",
                    xaxis_title="Feature Values",
                    yaxis_title="SHAP Values"
                )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating SHAP summary plot: {str(e)}")
            raise
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0
    ) -> go.Figure:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            
        Returns:
            Plotly figure object
        """
        try:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            # Get SHAP values for the sample
            sample_shap = self.shap_values[sample_idx]
            sample_features = X.iloc[sample_idx]
            
            # Create waterfall data
            waterfall_data = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Value': sample_shap,
                'Feature_Value': sample_features.values
            }).sort_values('SHAP_Value', ascending=False)
            
            # Calculate cumulative values
            waterfall_data['Cumulative'] = waterfall_data['SHAP_Value'].cumsum()
            
            # Create waterfall plot
            fig = go.Figure()
            
            # Add bars
            fig.add_trace(go.Waterfall(
                name="SHAP Values",
                orientation="h",
                measure=["relative"] * len(waterfall_data),
                x=waterfall_data['SHAP_Value'],
                textposition="outside",
                text=waterfall_data['Feature'],
                y=waterfall_data['Feature'],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title=f"SHAP Waterfall Plot - Sample {sample_idx}",
                xaxis_title="SHAP Value",
                yaxis_title="Features",
                showlegend=False,
                height=400 + len(waterfall_data) * 30
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating SHAP waterfall plot: {str(e)}")
            raise
    
    def plot_force(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0
    ) -> go.Figure:
        """
        Create SHAP force plot for a single prediction
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            
        Returns:
            Plotly figure object
        """
        try:
            if self.shap_values is None:
                self.calculate_shap_values(X)
            
            # Get SHAP values for the sample
            sample_shap = self.shap_values[sample_idx]
            sample_features = X.iloc[sample_idx]
            
            # Create force plot data
            force_data = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Value': sample_shap,
                'Feature_Value': sample_features.values
            })
            
            # Separate positive and negative contributions
            positive_contrib = force_data[force_data['SHAP_Value'] > 0]
            negative_contrib = force_data[force_data['SHAP_Value'] < 0]
            
            # Create subplot
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Positive Contributions', 'Negative Contributions'),
                vertical_spacing=0.1
            )
            
            # Positive contributions
            if not positive_contrib.empty:
                fig.add_trace(
                    go.Bar(
                        x=positive_contrib['SHAP_Value'],
                        y=positive_contrib['Feature'],
                        orientation='h',
                        marker_color='green',
                        name='Positive',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Negative contributions
            if not negative_contrib.empty:
                fig.add_trace(
                    go.Bar(
                        x=negative_contrib['SHAP_Value'],
                        y=negative_contrib['Feature'],
                        orientation='h',
                        marker_color='red',
                        name='Negative',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                title=f"SHAP Force Plot - Sample {sample_idx}",
                height=600,
                showlegend=False
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating SHAP force plot: {str(e)}")
            raise

class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) utilities"""
    
    def __init__(self):
        self.explainer = None
        self.feature_names = None
    
    def create_explainer(
        self,
        X_train: pd.DataFrame,
        model,
        mode: str = "classification"
    ) -> lime.lime_tabular.LimeTabularExplainer:
        """
        Create LIME explainer
        
        Args:
            X_train: Training features
            model: Trained machine learning model
            mode: 'classification' or 'regression'
            
        Returns:
            LIME explainer object
        """
        try:
            self.explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=['Class 0', 'Class 1'] if mode == 'classification' else None,
                mode=mode
            )
            
            self.feature_names = X_train.columns.tolist()
            logger.info(f"LIME explainer created for {mode} mode")
            return self.explainer
        
        except Exception as e:
            logger.error(f"Error creating LIME explainer: {str(e)}")
            raise
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain a single prediction using LIME
        
        Args:
            X: Input features
            sample_idx: Index of the sample to explain
            num_features: Number of top features to show
            
        Returns:
            Explanation dictionary
        """
        try:
            if self.explainer is None:
                raise ValueError("LIME explainer not initialized. Call create_explainer first.")
            
            sample = X.iloc[sample_idx]
            explanation = self.explainer.explain_instance(
                sample.values,
                lambda x: np.array([model.predict(x) for _ in range(len(x))]),
                num_features=num_features
            )
            
            # Extract explanation data
            exp_list = explanation.as_list()
            
            explanation_data = {
                'sample_idx': sample_idx,
                'prediction': explanation.predicted_value,
                'confidence': explanation.score,
                'features': exp_list,
                'feature_weights': dict(exp_list)
            }
            
            logger.info(f"LIME explanation generated for sample {sample_idx}")
            return explanation_data
        
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {str(e)}")
            raise
    
    def plot_explanation(
        self,
        explanation_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Create LIME explanation visualization
        
        Args:
            explanation_data: LIME explanation results
            
        Returns:
            Plotly figure object
        """
        try:
            features = explanation_data['features']
            
            # Create horizontal bar chart
            feature_names = [f[0] for f in features]
            weights = [f[1] for f in features]
            colors = ['green' if w > 0 else 'red' for w in weights]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weights,
                y=feature_names,
                orientation='h',
                marker_color=colors
            ))
            
            fig.update_layout(
                title=f"LIME Explanation - Sample {explanation_data['sample_idx']}",
                xaxis_title="Feature Weight",
                yaxis_title="Features",
                height=400 + len(features) * 30
            )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating LIME explanation plot: {str(e)}")
            raise

class FeatureImportanceAnalyzer:
    """Feature importance analysis utilities"""
    
    def __init__(self):
        self.importance_scores = {}
    
    def calculate_feature_importance(
        self,
        model,
        feature_names: List[str],
        method: str = "permutation"
    ) -> Dict[str, float]:
        """
        Calculate feature importance using various methods
        
        Args:
            model: Trained machine learning model
            feature_names: List of feature names
            method: Importance calculation method ('permutation', 'builtin', 'shap')
            
        Returns:
            Dictionary of feature importance scores
        """
        try:
            if method == "builtin":
                # Use model's built-in feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])  # For binary classification
                else:
                    raise ValueError("Model does not have built-in feature importance")
                
            elif method == "permutation":
                # Calculate permutation importance
                importance = self._calculate_permutation_importance(model, feature_names)
            
            else:
                raise ValueError(f"Unsupported importance method: {method}")
            
            # Create feature importance dictionary
            self.importance_scores = dict(zip(feature_names, importance))
            
            logger.info(f"Feature importance calculated using {method} method")
            return self.importance_scores
        
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise
    
    def _calculate_permutation_importance(
        self,
        model,
        feature_names: List[str],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        n_repeats: int = 10
    ) -> np.ndarray:
        """Calculate permutation importance"""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42
        )
        
        return result.importances_mean
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        plot_type: str = "bar"
    ) -> go.Figure:
        """
        Create feature importance visualization
        
        Args:
            top_n: Number of top features to display
            plot_type: Type of plot ('bar', 'horizontal')
            
        Returns:
            Plotly figure object
        """
        try:
            if not self.importance_scores:
                raise ValueError("No feature importance scores calculated")
            
            # Sort features by importance
            sorted_features = sorted(
                self.importance_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            fig = go.Figure()
            
            if plot_type == "bar":
                fig.add_trace(go.Bar(
                    x=feature_names,
                    y=importance_values,
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Features",
                    yaxis_title="Importance Score",
                    xaxis_tickangle=-45
                )
            
            elif plot_type == "horizontal":
                fig.add_trace(go.Bar(
                    x=importance_values,
                    y=feature_names,
                    orientation='h',
                    marker_color='steelblue'
                ))
                fig.update_layout(
                    title="Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Features",
                    height=400 + len(feature_names) * 20
                )
            
            return fig
        
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise

class ModelInterpretability:
    """Main class for model interpretability and explainability"""
    
    def __init__(self):
        self.shap_explainer = SHAPExplainer()
        self.lime_explainer = LIMEExplainer()
        self.feature_analyzer = FeatureImportanceAnalyzer()
    
    def generate_comprehensive_explanation(
        self,
        model,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Generate comprehensive model explanation using multiple methods
        
        Args:
            model: Trained machine learning model
            X_train: Training features
            X_test: Test features
            y_test: Test labels
            sample_idx: Index of sample to explain
            
        Returns:
            Comprehensive explanation dictionary
        """
        try:
            explanation_results = {}
            
            # SHAP analysis
            logger.info("Generating SHAP explanations...")
            self.shap_explainer.create_explainer(model, X_train)
            self.shap_explainer.calculate_shap_values(X_test)
            
            explanation_results['shap'] = {
                'summary_plot': self.shap_explainer.plot_summary(X_test),
                'waterfall_plot': self.shap_explainer.plot_waterfall(X_test, sample_idx),
                'force_plot': self.shap_explainer.plot_force(X_test, sample_idx)
            }
            
            # LIME analysis
            logger.info("Generating LIME explanations...")
            mode = 'classification' if len(y_test.unique()) < 10 else 'regression'
            self.lime_explainer.create_explainer(X_train, model, mode)
            lime_explanation = self.lime_explainer.explain_prediction(X_test, sample_idx)
            
            explanation_results['lime'] = {
                'explanation': lime_explanation,
                'plot': self.lime_explainer.plot_explanation(lime_explanation)
            }
            
            # Feature importance
            logger.info("Calculating feature importance...")
            feature_importance = self.feature_analyzer.calculate_feature_importance(
                model, X_train.columns.tolist()
            )
            
            explanation_results['feature_importance'] = {
                'scores': feature_importance,
                'plot': self.feature_analyzer.plot_feature_importance()
            }
            
            logger.info("Comprehensive explanation generated successfully")
            return explanation_results
        
        except Exception as e:
            logger.error(f"Error generating comprehensive explanation: {str(e)}")
            raise 