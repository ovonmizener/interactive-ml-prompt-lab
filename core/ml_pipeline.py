"""
ML Pipeline - Core machine learning functionality
Data loaders, training loops, model evaluation, and metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading and preprocessing utilities"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Preprocess data for machine learning
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            feature_columns: List of feature columns (if None, use all except target)
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing preprocessed data
        """
        try:
            # Select features
            if feature_columns is None:
                feature_columns = [col for col in df.columns if col != target_column]
            
            # Handle missing values
            df_clean = df.dropna(subset=[target_column])
            
            # Separate features and target
            X = df_clean[feature_columns].copy()
            y = df_clean[target_column].copy()
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                X[col] = self.label_encoder.fit_transform(X[col].astype(str))
            
            # Scale numerical features
            numerical_columns = X.select_dtypes(include=[np.number]).columns
            if len(numerical_columns) > 0:
                X[numerical_columns] = self.scaler.fit_transform(X[numerical_columns])
                self.is_fitted = True
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            result = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'feature_columns': feature_columns,
                'target_column': target_column,
                'categorical_columns': list(categorical_columns),
                'numerical_columns': list(numerical_columns)
            }
            
            logger.info(f"Data preprocessing completed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
            return result
        
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

class ModelTrainer:
    """Model training and evaluation utilities"""
    
    def __init__(self):
        self.models = {}
        self.training_history = {}
        self.best_model = None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available ML models and their configurations"""
        return {
            "Linear Regression": {
                "type": "regression",
                "class": LinearRegression,
                "description": "Simple linear regression for continuous targets",
                "hyperparameters": {
                    "fit_intercept": [True, False],
                    "normalize": [True, False]
                }
            },
            "Logistic Regression": {
                "type": "classification",
                "class": LogisticRegression,
                "description": "Linear classification for binary/multiclass problems",
                "hyperparameters": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            },
            "Random Forest": {
                "type": "both",
                "class": RandomForestClassifier,
                "description": "Ensemble method for both classification and regression",
                "hyperparameters": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            },
            "Support Vector Machine": {
                "type": "both",
                "class": SVC,
                "description": "Powerful algorithm for classification and regression",
                "hyperparameters": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["rbf", "linear", "poly"],
                    "gamma": ["scale", "auto"]
                }
            },
            "Neural Network": {
                "type": "both",
                "class": MLPClassifier,
                "description": "Multi-layer perceptron for complex patterns",
                "hyperparameters": {
                    "hidden_layer_sizes": [(50,), (100,), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "solver": ["adam", "sgd"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "max_iter": [200, 500]
                }
            }
        }
    
    def train_model(
        self,
        model_name: str,
        data: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train a machine learning model
        
        Args:
            model_name: Name of the model to train
            data: Preprocessed data dictionary
            hyperparameters: Model hyperparameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training results dictionary
        """
        try:
            # Get model configuration
            available_models = self.get_available_models()
            if model_name not in available_models:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_config = available_models[model_name]
            model_class = model_config["class"]
            
            # Determine if classification or regression
            y_train = data['y_train']
            if y_train.dtype in ['object', 'category'] or len(y_train.unique()) < 10:
                problem_type = "classification"
            else:
                problem_type = "regression"
            
            # Initialize model
            model = model_class(**hyperparameters)
            
            # Train model
            logger.info(f"Training {model_name} with {problem_type} problem type")
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_pred_train = model.predict(data['X_train'])
            y_pred_test = model.predict(data['X_test'])
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                data['y_train'], y_pred_train,
                data['y_test'], y_pred_test,
                problem_type
            )
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, data['X_train'], data['y_train'],
                cv=cv_folds, scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            
            # Store results
            training_result = {
                'model': model,
                'model_name': model_name,
                'problem_type': problem_type,
                'hyperparameters': hyperparameters,
                'metrics': metrics,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_date': datetime.now().isoformat()
            }
            
            self.models[model_name] = training_result
            
            logger.info(f"Model training completed. CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            return training_result
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def calculate_metrics(
        self,
        y_train_true: pd.Series,
        y_train_pred: np.ndarray,
        y_test_true: pd.Series,
        y_test_pred: np.ndarray,
        problem_type: str
    ) -> Dict[str, float]:
        """
        Calculate model performance metrics
        
        Args:
            y_train_true: True training labels
            y_train_pred: Predicted training labels
            y_test_true: True test labels
            y_test_pred: Predicted test labels
            problem_type: 'classification' or 'regression'
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if problem_type == "classification":
            # Classification metrics
            metrics['train_accuracy'] = accuracy_score(y_train_true, y_train_pred)
            metrics['test_accuracy'] = accuracy_score(y_test_true, y_test_pred)
            metrics['train_precision'] = precision_score(y_train_true, y_train_pred, average='weighted')
            metrics['test_precision'] = precision_score(y_test_true, y_test_pred, average='weighted')
            metrics['train_recall'] = recall_score(y_train_true, y_train_pred, average='weighted')
            metrics['test_recall'] = recall_score(y_test_true, y_test_pred, average='weighted')
            metrics['train_f1'] = f1_score(y_train_true, y_train_pred, average='weighted')
            metrics['test_f1'] = f1_score(y_test_true, y_test_pred, average='weighted')
        
        else:
            # Regression metrics
            metrics['train_mse'] = mean_squared_error(y_train_true, y_train_pred)
            metrics['test_mse'] = mean_squared_error(y_test_true, y_test_pred)
            metrics['train_mae'] = mean_absolute_error(y_train_true, y_train_pred)
            metrics['test_mae'] = mean_absolute_error(y_test_true, y_test_pred)
            metrics['train_r2'] = r2_score(y_train_true, y_train_pred)
            metrics['test_r2'] = r2_score(y_test_true, y_test_pred)
        
        return metrics
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        data: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            model_name: Name of the model
            data: Preprocessed data dictionary
            param_grid: Parameter grid for tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuning results dictionary
        """
        try:
            available_models = self.get_available_models()
            model_class = available_models[model_name]["class"]
            
            # Initialize base model
            base_model = model_class()
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='accuracy' if len(data['y_train'].unique()) < 10 else 'r2',
                n_jobs=-1
            )
            
            grid_search.fit(data['X_train'], data['y_train'])
            
            # Get best model
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Train best model on full training set
            training_result = self.train_model(
                model_name, data, best_params, cv_folds
            )
            
            tuning_result = {
                'best_params': best_params,
                'best_score': best_score,
                'cv_results': grid_search.cv_results_,
                'training_result': training_result
            }
            
            logger.info(f"Hyperparameter tuning completed. Best score: {best_score:.3f}")
            return tuning_result
        
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def save_model(self, model_name: str, file_path: str):
        """Save trained model to file"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model_data = self.models[model_name]
            joblib.dump(model_data, file_path)
            logger.info(f"Model saved to {file_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, file_path: str) -> Dict[str, Any]:
        """Load trained model from file"""
        try:
            model_data = joblib.load(file_path)
            model_name = model_data['model_name']
            self.models[model_name] = model_data
            logger.info(f"Model loaded from {file_path}")
            return model_data
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, model_name: str, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using trained model"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]['model']
            return model.predict(X)
        
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

class ModelEvaluator:
    """Model evaluation and comparison utilities"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def compare_models(self, models: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple models based on their performance
        
        Args:
            models: Dictionary of trained models
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for model_name, model_data in models.items():
            metrics = model_data['metrics']
            cv_mean = model_data['cv_mean']
            cv_std = model_data['cv_std']
            
            row = {
                'Model': model_name,
                'Problem Type': model_data['problem_type'],
                'CV Score': f"{cv_mean:.3f} (+/- {cv_std * 2:.3f})",
                'CV Mean': cv_mean,
                'CV Std': cv_std
            }
            
            # Add specific metrics based on problem type
            if model_data['problem_type'] == 'classification':
                row.update({
                    'Test Accuracy': f"{metrics['test_accuracy']:.3f}",
                    'Test Precision': f"{metrics['test_precision']:.3f}",
                    'Test Recall': f"{metrics['test_recall']:.3f}",
                    'Test F1': f"{metrics['test_f1']:.3f}"
                })
            else:
                row.update({
                    'Test R²': f"{metrics['test_r2']:.3f}",
                    'Test MSE': f"{metrics['test_mse']:.3f}",
                    'Test MAE': f"{metrics['test_mae']:.3f}"
                })
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, model_name: str, model_data: Dict[str, Any]) -> str:
        """
        Generate a detailed model evaluation report
        
        Args:
            model_name: Name of the model
            model_data: Model training results
            
        Returns:
            Formatted report string
        """
        report = f"""
# Model Evaluation Report: {model_name}

## Model Information
- **Model Type**: {model_data['model_name']}
- **Problem Type**: {model_data['problem_type']}
- **Training Date**: {model_data['training_date']}

## Hyperparameters
```json
{json.dumps(model_data['hyperparameters'], indent=2)}
```

## Performance Metrics
"""
        
        metrics = model_data['metrics']
        if model_data['problem_type'] == 'classification':
            report += f"""
- **Cross-Validation Score**: {model_data['cv_mean']:.3f} (+/- {model_data['cv_std'] * 2:.3f})
- **Test Accuracy**: {metrics['test_accuracy']:.3f}
- **Test Precision**: {metrics['test_precision']:.3f}
- **Test Recall**: {metrics['test_recall']:.3f}
- **Test F1-Score**: {metrics['test_f1']:.3f}
"""
        else:
            report += f"""
- **Cross-Validation R²**: {model_data['cv_mean']:.3f} (+/- {model_data['cv_std'] * 2:.3f})
- **Test R²**: {metrics['test_r2']:.3f}
- **Test MSE**: {metrics['test_mse']:.3f}
- **Test MAE**: {metrics['test_mae']:.3f}
"""
        
        return report 