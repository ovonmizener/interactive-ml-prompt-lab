"""
Unit tests for ML pipeline module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ml_pipeline import DataLoader, ModelTrainer, ModelEvaluator

class TestDataLoader:
    """Test cases for DataLoader class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_loader = DataLoader()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 1]
        })
    
    def test_load_data_csv(self):
        """Test loading CSV data"""
        # Create temporary CSV file
        temp_file = "temp_test.csv"
        self.sample_data.to_csv(temp_file, index=False)
        
        try:
            result = self.data_loader.load_data(temp_file)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5
            assert list(result.columns) == ['feature1', 'feature2', 'target']
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def test_preprocess_data(self):
        """Test data preprocessing"""
        result = self.data_loader.preprocess_data(
            self.sample_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        assert 'X_train' in result
        assert 'X_test' in result
        assert 'y_train' in result
        assert 'y_test' in result
        assert 'feature_columns' in result
        assert 'target_column' in result
        
        # Check shapes
        assert len(result['X_train']) + len(result['X_test']) == len(self.sample_data)
        assert len(result['y_train']) + len(result['y_test']) == len(self.sample_data)
    
    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing with missing values"""
        # Add missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'feature1'] = np.nan
        
        result = self.data_loader.preprocess_data(
            data_with_missing,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Should handle missing values gracefully
        assert 'X_train' in result
        assert 'X_test' in result

class TestModelTrainer:
    """Test cases for ModelTrainer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.trainer = ModelTrainer()
        
        # Create sample data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        self.X_test = pd.DataFrame({
            'feature1': [6, 7],
            'feature2': [0.6, 0.7]
        })
        self.y_train = pd.Series([0, 1, 0, 1, 1])
        self.y_test = pd.Series([1, 0])
        
        self.data = {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'feature_columns': ['feature1', 'feature2'],
            'target_column': 'target'
        }
    
    def test_get_available_models(self):
        """Test getting available models"""
        models = self.trainer.get_available_models()
        
        assert isinstance(models, dict)
        assert 'Linear Regression' in models
        assert 'Random Forest' in models
        assert 'Support Vector Machine' in models
        assert 'Neural Network' in models
        
        # Check model structure
        for model_name, model_info in models.items():
            assert 'type' in model_info
            assert 'description' in model_info
            assert 'hyperparameters' in model_info
    
    def test_get_model_hyperparameters(self):
        """Test getting model hyperparameters"""
        hyperparams = self.trainer.get_model_hyperparameters('Linear Regression')
        assert isinstance(hyperparams, dict)
        assert 'fit_intercept' in hyperparams
    
    @patch('core.ml_pipeline.LinearRegression')
    def test_train_model(self, mock_linear_regression):
        """Test model training"""
        # Mock the model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0.5, 0.8])
        mock_linear_regression.return_value = mock_model
        
        result = self.trainer.train_model(
            'Linear Regression',
            self.data,
            {'fit_intercept': True}
        )
        
        assert isinstance(result, dict)
        assert 'model' in result
        assert 'model_name' in result
        assert 'problem_type' in result
        assert 'metrics' in result
        assert 'cv_scores' in result
    
    def test_calculate_metrics_classification(self):
        """Test metric calculation for classification"""
        y_true = pd.Series([0, 1, 0, 1, 1])
        y_pred_train = np.array([0, 1, 0, 1, 1])
        y_pred_test = np.array([0, 1])
        y_test = pd.Series([0, 1])
        
        metrics = self.trainer.calculate_metrics(
            y_true, y_pred_train, y_test, y_pred_test, 'classification'
        )
        
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'train_precision' in metrics
        assert 'test_precision' in metrics
        assert 'train_recall' in metrics
        assert 'test_recall' in metrics
        assert 'train_f1' in metrics
        assert 'test_f1' in metrics
    
    def test_calculate_metrics_regression(self):
        """Test metric calculation for regression"""
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred_train = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        y_pred_test = np.array([2.1, 3.9])
        y_test = pd.Series([2.0, 4.0])
        
        metrics = self.trainer.calculate_metrics(
            y_true, y_pred_train, y_test, y_pred_test, 'regression'
        )
        
        assert 'train_mse' in metrics
        assert 'test_mse' in metrics
        assert 'train_mae' in metrics
        assert 'test_mae' in metrics
        assert 'train_r2' in metrics
        assert 'test_r2' in metrics

class TestModelEvaluator:
    """Test cases for ModelEvaluator class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.evaluator = ModelEvaluator()
        
        # Create mock models
        self.mock_models = {
            'Model1': {
                'model_name': 'Linear Regression',
                'problem_type': 'regression',
                'metrics': {
                    'test_r2': 0.85,
                    'test_mse': 0.15
                },
                'cv_mean': 0.83,
                'cv_std': 0.05
            },
            'Model2': {
                'model_name': 'Random Forest',
                'problem_type': 'classification',
                'metrics': {
                    'test_accuracy': 0.92,
                    'test_f1': 0.91
                },
                'cv_mean': 0.90,
                'cv_std': 0.03
            }
        }
    
    def test_compare_models(self):
        """Test model comparison"""
        comparison_df = self.evaluator.compare_models(self.mock_models)
        
        assert isinstance(comparison_df, pd.DataFrame)
        assert len(comparison_df) == 2
        assert 'Model' in comparison_df.columns
        assert 'Problem Type' in comparison_df.columns
        assert 'CV Score' in comparison_df.columns
    
    def test_generate_report(self):
        """Test report generation"""
        model_data = self.mock_models['Model1']
        report = self.evaluator.generate_report('Model1', model_data)
        
        assert isinstance(report, str)
        assert 'Model Evaluation Report' in report
        assert 'Model1' in report
        assert 'Linear Regression' in report
        assert 'regression' in report

# Integration tests
class TestIntegration:
    """Integration tests for the ML pipeline"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.data_loader = DataLoader()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        # Create comprehensive test data
        np.random.seed(42)
        n_samples = 100
        
        self.test_data = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        })
    
    def test_end_to_end_pipeline(self):
        """Test complete ML pipeline from data loading to evaluation"""
        # 1. Preprocess data
        processed_data = self.data_loader.preprocess_data(
            self.test_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # 2. Train model
        training_result = self.trainer.train_model(
            'Random Forest',
            processed_data,
            {'n_estimators': 10, 'max_depth': 5}
        )
        
        # 3. Store model
        self.trainer.models['test_model'] = training_result
        
        # 4. Compare models
        comparison_df = self.evaluator.compare_models(self.trainer.models)
        
        # Assertions
        assert len(processed_data['X_train']) > 0
        assert len(processed_data['X_test']) > 0
        assert training_result['model_name'] == 'Random Forest'
        assert 'test_model' in self.trainer.models
        assert len(comparison_df) >= 1
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning functionality"""
        # Preprocess data
        processed_data = self.data_loader.preprocess_data(
            self.test_data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [5, 10],
            'max_depth': [3, 5]
        }
        
        # Test hyperparameter tuning
        tuning_result = self.trainer.hyperparameter_tuning(
            'Random Forest',
            processed_data,
            param_grid,
            cv_folds=3
        )
        
        assert 'best_params' in tuning_result
        assert 'best_score' in tuning_result
        assert 'training_result' in tuning_result

# Test utilities
def create_test_data(n_samples=100, n_features=5, n_classes=2):
    """Create test data for unit tests"""
    np.random.seed(42)
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(range(n_classes), n_samples)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

if __name__ == "__main__":
    pytest.main([__file__]) 