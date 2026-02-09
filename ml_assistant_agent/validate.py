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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from . import config
from .goal_detector import detect_ml_task, goal_detector
import logging
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)


def advanced_validate(file_path: str, goal: str, target_column: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform advanced ML validation with automatic goal detection and model recommendations
    
    Args:
        file_path: Path to the CSV file
        goal: User's ML objective description
        target_column: Name of the target column (optional, will be auto-detected if not provided)
        
    Returns:
        Dictionary containing detailed validation results with goal-based recommendations
    """
    try:
        logger.info(f"Starting advanced validation for {file_path}")
        logger.info(f"User goal: '{goal[:200]}...'")
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {df.shape} shape")
        
        # Perform goal and dataset analysis
        goal_analysis = detect_ml_task(goal, df, target_column)
        
        # Extract recommendations
        recommendations = goal_analysis['recommendations']
        final_task = recommendations['final_task_type']
        requires_target = recommendations['requires_target_column']
        
        logger.info(f"Detected task type: {final_task}")
        
        # Handle target column based on task type
        if requires_target:
            if target_column is None:
                # Try to auto-detect target column
                target_column = _auto_detect_target_column(df, final_task)
                if target_column is None:
                    return {
                        "error": "Supervised learning task detected but no target column specified. Please provide target_column parameter.",
                        "goal_analysis": goal_analysis,
                        "available_columns": list(df.columns)
                    }
            
            if target_column not in df.columns:
                return {
                    "error": f"Target column '{target_column}' not found. Available columns: {list(df.columns)}",
                    "goal_analysis": goal_analysis
                }
        
        # Build comprehensive result starting with goal analysis
        result = {
            "goal_analysis": goal_analysis,
            "detected_task_type": final_task,
            "learning_type": recommendations['learning_type'],
            "confidence": recommendations['confidence'],
            "reasoning": recommendations['reasoning'],
            "algorithm_recommendations": recommendations['algorithms']
        }
        
        # Perform task-specific validation
        if final_task in ['classification', 'regression']:
            result.update(_perform_supervised_validation(df, target_column, final_task))
        elif final_task == 'clustering':
            result.update(_perform_clustering_validation(df))
        elif final_task == 'anomaly_detection':
            result.update(_perform_anomaly_validation(df))
        else:
            result.update(_perform_general_validation(df))
        
        logger.info(f"Advanced validation completed for {file_path}")
        return result
        
    except Exception as e:
        error_msg = f"Advanced validation failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}


def _auto_detect_target_column(df: pd.DataFrame, task_type: str) -> Optional[str]:
    """
    Attempt to automatically detect the target column based on dataset characteristics
    
    Args:
        df: Input dataframe
        task_type: Detected ML task type
        
    Returns:
        Name of likely target column or None if not found
    """
    # Common target column names
    target_keywords = {
        'classification': ['target', 'class', 'label', 'category', 'outcome', 'result', 
                         'prediction', 'churn', 'fraud', 'spam', 'diagnosis'],
        'regression': ['target', 'value', 'price', 'cost', 'amount', 'score', 'rating',
                      'revenue', 'sales', 'salary', 'income', 'prediction']
    }
    
    keywords = target_keywords.get(task_type, target_keywords['classification'])
    
    # Check for exact matches first
    for col in df.columns:
        if col.lower() in keywords:
            logger.info(f"Auto-detected target column: {col} (exact match)")
            return col
    
    # Check for partial matches
    for col in df.columns:
        col_lower = col.lower()
        for keyword in keywords:
            if keyword in col_lower or col_lower in keyword:
                logger.info(f"Auto-detected target column: {col} (partial match: {keyword})")
                return col
    
    # Check for numeric columns that might be targets (for regression)
    if task_type == 'regression':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            logger.info(f"Auto-detected target column: {numeric_cols[0]} (only numeric column)")
            return numeric_cols[0]
    
    # Check for binary columns (for classification)
    if task_type == 'classification':
        for col in df.columns:
            if df[col].nunique() == 2:
                logger.info(f"Auto-detected target column: {col} (binary column)")
                return col
    
    logger.warning("Could not auto-detect target column")
    return None


def _perform_supervised_validation(df: pd.DataFrame, target_column: str, task_type: str) -> Dict[str, Any]:
    """
    Perform validation for supervised learning tasks (classification/regression)
    
    Args:
        df: Input dataframe
        target_column: Target column name
        task_type: 'classification' or 'regression'
        
    Returns:
        Validation results dictionary
    """
    # Clean data and separate features/target
    df_clean = df.dropna(subset=[target_column]).copy()
    
    if len(df_clean) < config.MIN_SAMPLES_FOR_VALIDATION:
        error_msg = f"Insufficient data: {len(df_clean)} samples (minimum: {config.MIN_SAMPLES_FOR_VALIDATION})"
        return {"error": error_msg}
    
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]
    
    # Preprocess features
    X_processed, preprocessing_info = advanced_preprocessing(X)
    
    if X_processed.shape[1] < config.MIN_FEATURES_FOR_VALIDATION:
        error_msg = f"Insufficient features after preprocessing: {X_processed.shape[1]}"
        return {"error": error_msg}
    
    # Task-specific validation
    if task_type == 'classification':
        unique_targets = y.nunique(dropna=True)
        validation_results = perform_classification(X_processed, y, unique_targets)
    else:  # regression
        validation_results = perform_regression(X_processed, y)
    
    # Add data summary
    validation_results.update({
        "data_summary": {
            "original_shape": df.shape,
            "clean_shape": df_clean.shape,
            "target_column": target_column,
            "target_missing_count": df[target_column].isnull().sum(),
            "features_original": len(X.columns),
            "features_processed": X_processed.shape[1]
        },
        "preprocessing": preprocessing_info
    })
    
    return validation_results


def _perform_clustering_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform validation for clustering tasks
    
    Args:
        df: Input dataframe
        
    Returns:
        Clustering validation results
    """
    # Preprocess data for clustering
    X_processed, preprocessing_info = advanced_preprocessing(df)
    
    if X_processed.shape[1] < 2:
        return {"error": "Insufficient features for clustering (minimum: 2)"}
    
    results = {
        "task_type": "clustering",
        "data_summary": {
            "shape": df.shape,
            "features_processed": X_processed.shape[1]
        },
        "preprocessing": preprocessing_info,
        "clustering_analysis": {}
    }
    
    # Perform K-means clustering with different k values
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        k_values = range(2, min(11, len(X_processed) // 2))
        clustering_scores = {}
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=config.DEFAULT_RANDOM_STATE)
            cluster_labels = kmeans.fit_predict(X_processed)
            silhouette_avg = silhouette_score(X_processed, cluster_labels)
            clustering_scores[k] = {
                'silhouette_score': float(silhouette_avg),
                'inertia': float(kmeans.inertia_)
            }
        
        # Find optimal k
        best_k = max(clustering_scores.keys(), key=lambda k: clustering_scores[k]['silhouette_score'])
        
        results["clustering_analysis"] = {
            "k_means_results": clustering_scores,
            "recommended_clusters": int(best_k),
            "best_silhouette_score": clustering_scores[best_k]['silhouette_score']
        }
        
    except Exception as e:
        logger.error(f"Clustering analysis failed: {e}")
        results["clustering_analysis"]["error"] = str(e)
    
    return results


def _perform_anomaly_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform validation for anomaly detection tasks
    
    Args:
        df: Input dataframe
        
    Returns:
        Anomaly detection validation results
    """
    # Preprocess data
    X_processed, preprocessing_info = advanced_preprocessing(df)
    
    results = {
        "task_type": "anomaly_detection",
        "data_summary": {
            "shape": df.shape,
            "features_processed": X_processed.shape[1]
        },
        "preprocessing": preprocessing_info,
        "anomaly_analysis": {}
    }
    
    try:
        from sklearn.ensemble import IsolationForest
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=config.DEFAULT_RANDOM_STATE)
        outlier_labels = iso_forest.fit_predict(X_processed)
        
        n_outliers = np.sum(outlier_labels == -1)
        outlier_percentage = (n_outliers / len(X_processed)) * 100
        
        results["anomaly_analysis"] = {
            "method": "Isolation Forest",
            "total_samples": len(X_processed),
            "detected_outliers": int(n_outliers),
            "outlier_percentage": float(outlier_percentage),
            "contamination_rate": 0.1
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection analysis failed: {e}")
        results["anomaly_analysis"]["error"] = str(e)
    
    return results


def _perform_general_validation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform general dataset validation when task type is unclear
    
    Args:
        df: Input dataframe
        
    Returns:
        General validation results
    """
    X_processed, preprocessing_info = advanced_preprocessing(df)
    
    return {
        "task_type": "general_analysis",
        "data_summary": {
            "shape": df.shape,
            "features_processed": X_processed.shape[1]
        },
        "preprocessing": preprocessing_info,
        "note": "General dataset analysis performed. Please provide more specific goals for targeted ML recommendations."
    }


def simple_validate(file_path: str, target_column: str) -> Dict[str, Any]:
    """
    Legacy function - redirects to advanced_validate with generic goal
    
    Args:
        file_path: Path to the CSV file
        target_column: Name of the target column
        
    Returns:
        Dictionary containing detailed validation results
    """
    generic_goal = f"Analyze the dataset and build a model to predict {target_column}"
    return advanced_validate(file_path, generic_goal, target_column)
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
