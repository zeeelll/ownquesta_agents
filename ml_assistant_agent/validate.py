from typing import Any, Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, classification_report, r2_score, mean_squared_error,
    precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, explained_variance_score, roc_auc_score
)
from sklearn.impute import SimpleImputer
from . import config
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


def simple_validate(file_path: str, target_column: str) -> Dict[str, Any]:
    """
    Perform comprehensive ML validation with advanced preprocessing and multiple models
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        
    Returns:
        Dictionary containing detailed validation results
    """
    try:
        logger.info(f"Starting advanced validation for {file_path} with target: {target_column}")
        
        # Load and validate data
        df = pd.read_csv(file_path)
        
        if target_column not in df.columns:
            error_msg = f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Initial data analysis
        original_shape = df.shape
        target_missing = df[target_column].isnull().sum()
        
        # Handle missing target values
        df_clean = df.dropna(subset=[target_column]).copy()
        
        if len(df_clean) < config.MIN_SAMPLES_FOR_VALIDATION:
            error_msg = f"Insufficient data: {len(df_clean)} samples (minimum: {config.MIN_SAMPLES_FOR_VALIDATION})"
            logger.error(error_msg)
            return {"error": error_msg}
        
        X = df_clean.drop(columns=[target_column])
        y = df_clean[target_column]

        # Comprehensive preprocessing
        X_processed, preprocessing_info = advanced_preprocessing(X)
        
        if X_processed.shape[1] < config.MIN_FEATURES_FOR_VALIDATION:
            error_msg = f"Insufficient features after preprocessing: {X_processed.shape[1]} (minimum: {config.MIN_FEATURES_FOR_VALIDATION})"
            logger.error(error_msg)
            return {"error": error_msg}

        # Determine task type and preprocess target
        unique_targets = y.nunique(dropna=True)
        task_type = "classification" if unique_targets <= config.MAX_CATEGORICAL_UNIQUE else "regression"
        
        # Build comprehensive result
        result: Dict[str, Any] = {
            "task_type": task_type,
            "data_summary": {
                "original_shape": original_shape,
                "clean_shape": df_clean.shape,
                "target_missing_count": int(target_missing),
                "samples_used": len(df_clean),
                "features_original": len(X.columns),
                "features_processed": X_processed.shape[1],
                "target_unique_values": int(unique_targets)
            },
            "preprocessing": preprocessing_info
        }

        logger.info(f"Task type: {task_type}, Samples: {len(df_clean)}, Features: {X_processed.shape[1]}")

        if task_type == "classification":
            # Classification workflow
            classification_results = perform_classification(X_processed, y, unique_targets)
            result.update(classification_results)
            
        else:
            # Regression workflow
            regression_results = perform_regression(X_processed, y)
            result.update(regression_results)

        # Add configuration information
        result["configuration"] = {
            "test_size": config.DEFAULT_TEST_SIZE,
            "random_state": config.DEFAULT_RANDOM_STATE,
            "cross_validation_folds": 5,
            "scaling_method": "robust",
            "imputation_strategy": "median for numeric, mode for categorical"
        }

        logger.info(f"Advanced validation completed for {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def advanced_preprocessing(X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Advanced preprocessing pipeline with comprehensive feature engineering
    
    Args:
        X: Feature dataframe
        
    Returns:
        Tuple of (processed_features, preprocessing_info)
    """
    preprocessing_info = {
        "original_features": list(X.columns),
        "steps_applied": [],
        "features_removed": [],
        "features_created": []
    }
    
    X_processed = X.copy()
    
    # 1. Remove constant columns
    constant_cols = X_processed.columns[X_processed.nunique() <= 1]
    if len(constant_cols) > 0:
        X_processed = X_processed.drop(columns=constant_cols)
        preprocessing_info["features_removed"].extend(constant_cols.tolist())
        preprocessing_info["steps_applied"].append(f"Removed {len(constant_cols)} constant columns")
    
    # 2. Handle missing values intelligently
    numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
    
    missing_info = {}
    
    # Numeric imputation
    if len(numeric_cols) > 0:
        numeric_imputer = SimpleImputer(strategy='median')
        X_processed[numeric_cols] = numeric_imputer.fit_transform(X_processed[numeric_cols])
        missing_info["numeric"] = f"Imputed with median for {len(numeric_cols)} columns"
        preprocessing_info["steps_applied"].append(missing_info["numeric"])
    
    # Categorical imputation
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            mode_value = X_processed[col].mode().iloc[0] if not X_processed[col].mode().empty else "Unknown"
            X_processed[col] = X_processed[col].fillna(mode_value)
        missing_info["categorical"] = f"Imputed with mode for {len(categorical_cols)} columns"
        preprocessing_info["steps_applied"].append(missing_info["categorical"])
    
    # 3. Encode categorical variables
    categorical_encoded = []
    for col in categorical_cols:
        if X_processed[col].nunique() <= config.MAX_CATEGORICAL_UNIQUE:
            # One-hot encode low cardinality categoricals
            dummies = pd.get_dummies(X_processed[col], prefix=col, drop_first=True)
            X_processed = pd.concat([X_processed.drop(columns=[col]), dummies], axis=1)
            categorical_encoded.append(col)
            preprocessing_info["features_created"].extend(dummies.columns.tolist())
        else:
            # Label encode high cardinality categoricals
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            categorical_encoded.append(col)
    
    if categorical_encoded:
        preprocessing_info["steps_applied"].append(f"Encoded {len(categorical_encoded)} categorical columns")
    
    # 4. Scale numeric features
    current_numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
    if len(current_numeric_cols) > 0:
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_processed[current_numeric_cols] = scaler.fit_transform(X_processed[current_numeric_cols])
        preprocessing_info["steps_applied"].append(f"Scaled {len(current_numeric_cols)} numeric features with RobustScaler")
    
    # 5. Feature selection based on variance
    low_variance_threshold = 0.01
    numeric_features = X_processed.select_dtypes(include=[np.number])
    if len(numeric_features.columns) > 0:
        variances = numeric_features.var()
        low_variance_cols = variances[variances < low_variance_threshold].index
        if len(low_variance_cols) > 0:
            X_processed = X_processed.drop(columns=low_variance_cols)
            preprocessing_info["features_removed"].extend(low_variance_cols.tolist())
            preprocessing_info["steps_applied"].append(f"Removed {len(low_variance_cols)} low-variance features")
    
    preprocessing_info["final_features"] = list(X_processed.columns)
    preprocessing_info["summary"] = {
        "original_count": len(X.columns),
        "final_count": len(X_processed.columns),
        "removed_count": len(preprocessing_info["features_removed"]),
        "created_count": len(preprocessing_info["features_created"])
    }
    
    return X_processed, preprocessing_info


def perform_classification(X: pd.DataFrame, y: pd.Series, unique_targets: int) -> Dict[str, Any]:
    """
    Perform comprehensive classification with multiple models and metrics
    """
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=config.DEFAULT_TEST_SIZE,
        random_state=config.DEFAULT_RANDOM_STATE,
        stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
    )
    
    # Train multiple models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=config.LOGISTIC_REGRESSION_MAX_ITER,
            random_state=config.DEFAULT_RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    results = {}
    best_model = None
    best_score = 0
    
    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
            
            # ROC AUC for binary classification
            roc_auc = None
            if unique_targets == 2 and y_pred_proba is not None:
                try:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                except Exception:
                    pass
            
            model_results = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "cross_val_mean": float(cv_scores.mean()),
                "cross_val_std": float(cv_scores.std()),
                "roc_auc": float(roc_auc) if roc_auc else None,
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                model_results["feature_importance"] = {k: float(v) for k, v in 
                                                     sorted(feature_importance.items(), 
                                                           key=lambda x: x[1], reverse=True)[:20]}
            
            results[model_name] = model_results
            
            # Track best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = model_name
                
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}
    
    return {
        "models": results,
        "best_model": best_model,
        "target_labels": le.classes_.tolist() if hasattr(le, 'classes_') else [],
        "class_distribution": pd.Series(y_encoded).value_counts().to_dict(),
        "split_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_class_dist": pd.Series(y_train).value_counts().to_dict(),
            "test_class_dist": pd.Series(y_test).value_counts().to_dict()
        }
    }


def perform_regression(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Perform comprehensive regression with multiple models and metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.DEFAULT_TEST_SIZE,
        random_state=config.DEFAULT_RANDOM_STATE
    )
    
    # Train multiple models
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=config.DEFAULT_RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    results = {}
    best_model = None
    best_score = float('-inf')
    
    for model_name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            explained_var = explained_variance_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Calculate percentage errors
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 1e-10))) * 100
            
            model_results = {
                "r2_score": float(r2),
                "mean_squared_error": float(mse),
                "root_mean_squared_error": float(rmse),
                "mean_absolute_error": float(mae),
                "mean_absolute_percentage_error": float(mape),
                "explained_variance_score": float(explained_var),
                "cross_val_mean": float(cv_scores.mean()),
                "cross_val_std": float(cv_scores.std())
            }
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                model_results["feature_importance"] = {k: float(v) for k, v in 
                                                     sorted(feature_importance.items(), 
                                                           key=lambda x: x[1], reverse=True)[:20]}
            
            results[model_name] = model_results
            
            # Track best model
            if r2 > best_score:
                best_score = r2
                best_model = model_name
                
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            results[model_name] = {"error": str(e)}
    
    return {
        "models": results,
        "best_model": best_model,
        "target_statistics": {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": float(y.mean()),
            "median": float(y.median()),
            "std": float(y.std()),
            "skewness": float(y.skew()),
            "kurtosis": float(y.kurtosis())
        },
        "split_info": {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_target_mean": float(y_train.mean()),
            "test_target_mean": float(y_test.mean())
        }
    }
