"""
Advanced Goal Detection and Model Type Classification
Automatically detects ML task type based on user goals and dataset characteristics
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class GoalDetector:
    """
    Advanced ML Goal Detection System
    Analyzes user goals and dataset to determine optimal ML approach
    """
    
    def __init__(self):
        # Classification keywords and patterns
        self.classification_patterns = {
            'predict_category': [
                r'predict\s+(category|class|type|label)',
                r'classify\s+(customers|users|products|items)',
                r'(binary|multi-class)\s+classification',
                r'detect\s+(fraud|spam|anomal)',
                r'diagnos[ei]s',
                r'categoriz[ei]',
            ],
            'outcome_prediction': [
                r'will\s+(customer|user)\s+(buy|purchase|churn|leave)',
                r'(success|failure|pass|fail)',
                r'(approve|reject|accept|deny)',
                r'(positive|negative|sentiment)',
            ],
            'binary_classification': [
                r'(yes|no)\s+prediction',
                r'(true|false)\s+classification',
                r'binary\s+outcome',
                r'(pass|fail)\s+prediction',
            ]
        }
        
        # Regression keywords and patterns
        self.regression_patterns = {
            'predict_value': [
                r'predict\s+(price|cost|value|amount|salary|revenue)',
                r'forecast\s+(sales|demand|consumption)',
                r'estimate\s+(time|duration|distance)',
                r'calculate\s+(score|rating|index)',
            ],
            'continuous_prediction': [
                r'how\s+(much|many|long|far)',
                r'numeric\s+prediction',
                r'continuous\s+value',
                r'regression\s+(analysis|model)',
            ]
        }
        
        # Clustering keywords and patterns  
        self.clustering_patterns = {
            'group_discovery': [
                r'(group|cluster|segment)\s+(customers|users|products)',
                r'find\s+(patterns|segments|groups)',
                r'unsupervised\s+(learning|clustering)',
                r'market\s+segmentation',
            ],
            'pattern_discovery': [
                r'discover\s+(hidden|latent)\s+patterns',
                r'identify\s+(customer|user)\s+segments',
                r'natural\s+groupings',
                r'cluster\s+analysis',
            ]
        }
        
        # Anomaly detection patterns
        self.anomaly_patterns = {
            'outlier_detection': [
                r'detect\s+(outliers|anomalies|unusual)',
                r'find\s+(abnormal|irregular)\s+patterns',
                r'fraud\s+detection',
                r'anomaly\s+detection',
            ]
        }
        
        # Time series patterns
        self.timeseries_patterns = {
            'time_prediction': [
                r'forecast\s+(future|next|upcoming)',
                r'time\s+series\s+(prediction|analysis)',
                r'predict\s+(trends|seasonality)',
                r'temporal\s+(pattern|analysis)',
            ]
        }

    def analyze_goal(self, user_goal: str) -> Dict[str, Any]:
        """
        Analyze user goal text to extract ML task intent
        
        Args:
            user_goal: User's description of their ML objective
            
        Returns:
            Dictionary containing detected task type and confidence
        """
        goal_lower = user_goal.lower().strip()
        
        # Initialize scores for different task types
        task_scores = {
            'classification': 0,
            'regression': 0, 
            'clustering': 0,
            'anomaly_detection': 0,
            'time_series': 0
        }
        
        detected_keywords = []
        
        # Score classification patterns
        for category, patterns in self.classification_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, goal_lower))
                if matches > 0:
                    task_scores['classification'] += matches * 2
                    detected_keywords.append(f"classification:{category}")
        
        # Score regression patterns
        for category, patterns in self.regression_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, goal_lower))
                if matches > 0:
                    task_scores['regression'] += matches * 2
                    detected_keywords.append(f"regression:{category}")
        
        # Score clustering patterns
        for category, patterns in self.clustering_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, goal_lower))
                if matches > 0:
                    task_scores['clustering'] += matches * 2
                    detected_keywords.append(f"clustering:{category}")
        
        # Score anomaly detection patterns
        for category, patterns in self.anomaly_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, goal_lower))
                if matches > 0:
                    task_scores['anomaly_detection'] += matches * 2
                    detected_keywords.append(f"anomaly:{category}")
        
        # Score time series patterns
        for category, patterns in self.timeseries_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, goal_lower))
                if matches > 0:
                    task_scores['time_series'] += matches * 2
                    detected_keywords.append(f"timeseries:{category}")
        
        # Additional heuristics
        if any(word in goal_lower for word in ['supervised', 'labeled', 'target', 'dependent']):
            # Likely supervised learning
            if any(word in goal_lower for word in ['category', 'class', 'type', 'classify']):
                task_scores['classification'] += 1
            elif any(word in goal_lower for word in ['predict', 'value', 'amount', 'score']):
                task_scores['regression'] += 1
        
        if any(word in goal_lower for word in ['unsupervised', 'unlabeled', 'hidden', 'discover']):
            task_scores['clustering'] += 1.5
            
        # Determine primary task type
        if sum(task_scores.values()) == 0:
            # No clear patterns detected - use generic approach
            primary_task = 'classification'  # Default fallback
            confidence = 0.3
            reasoning = "No clear ML task patterns detected. Defaulting to classification for analysis."
        else:
            primary_task = max(task_scores, key=task_scores.get)
            max_score = task_scores[primary_task]
            total_score = sum(task_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0
            reasoning = f"Detected {primary_task} based on goal analysis"
        
        return {
            'primary_task': primary_task,
            'confidence': confidence,
            'task_scores': task_scores,
            'detected_keywords': detected_keywords,
            'reasoning': reasoning,
            'is_supervised': primary_task in ['classification', 'regression'],
            'is_unsupervised': primary_task in ['clustering', 'anomaly_detection'],
            'requires_target': primary_task in ['classification', 'regression']
        }

    def analyze_dataset_characteristics(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze dataset characteristics to inform task type detection
        
        Args:
            df: Input dataframe
            target_col: Target column name (if provided)
            
        Returns:
            Dictionary with dataset analysis results
        """
        analysis = {
            'shape': df.shape,
            'n_features': len(df.columns) - (1 if target_col else 0),
            'feature_types': {},
            'missing_data': {},
            'target_analysis': None
        }
        
        # Analyze feature types
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_features = df.select_dtypes(include=['datetime']).columns.tolist()
        
        analysis['feature_types'] = {
            'numeric': len(numeric_features),
            'categorical': len(categorical_features), 
            'datetime': len(datetime_features),
            'numeric_features': numeric_features,
            'categorical_features': categorical_features,
            'datetime_features': datetime_features
        }
        
        # Missing data analysis
        missing_counts = df.isnull().sum()
        analysis['missing_data'] = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentage': (missing_counts / len(df) * 100).to_dict()
        }
        
        # Target column analysis (if provided)
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            unique_values = target_series.nunique()
            
            analysis['target_analysis'] = {
                'column_name': target_col,
                'unique_values': unique_values,
                'total_samples': len(target_series),
                'missing_count': df[target_col].isnull().sum(),
                'data_type': str(target_series.dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(target_series),
                'suggested_task': self._suggest_task_from_target(target_series)
            }
            
            # Add value distribution for categorical targets
            if unique_values <= 20:
                analysis['target_analysis']['value_distribution'] = target_series.value_counts().to_dict()
        
        return analysis

    def _suggest_task_from_target(self, target_series: pd.Series) -> str:
        """
        Suggest ML task based on target variable characteristics
        
        Args:
            target_series: Target variable data
            
        Returns:
            Suggested task type string
        """
        unique_values = target_series.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        
        if is_numeric:
            if unique_values <= 2:
                return 'binary_classification'
            elif unique_values <= 10:
                return 'multiclass_classification'
            else:
                # Check if values are actually continuous
                if target_series.dtype in ['float64', 'float32']:
                    return 'regression'
                else:
                    # Integer values - could be ordinal classification or regression
                    value_range = target_series.max() - target_series.min()
                    if value_range / unique_values < 2:  # Dense integer sequence
                        return 'regression'
                    else:
                        return 'ordinal_classification'
        else:
            # Non-numeric target
            if unique_values <= 2:
                return 'binary_classification'
            else:
                return 'multiclass_classification'

    def combine_goal_and_data_analysis(self, goal_analysis: Dict, data_analysis: Dict) -> Dict[str, Any]:
        """
        Combine goal analysis and dataset analysis to make final ML task recommendation
        
        Args:
            goal_analysis: Results from analyze_goal()
            data_analysis: Results from analyze_dataset_characteristics()
            
        Returns:
            Combined analysis with final recommendations
        """
        combined = {
            'goal_analysis': goal_analysis,
            'data_analysis': data_analysis,
            'recommendations': {}
        }
        
        # Get goal-based and data-based suggestions
        goal_task = goal_analysis['primary_task']
        goal_confidence = goal_analysis['confidence']
        
        # If we have target analysis, consider data-based suggestion
        data_task = None
        data_confidence = 0.7  # Default confidence for data-based analysis
        
        if data_analysis['target_analysis']:
            data_task = data_analysis['target_analysis']['suggested_task']
        
        # Determine final recommendation
        if goal_confidence >= 0.7:
            # High confidence in goal analysis
            final_task = goal_task
            reasoning = f"High confidence goal analysis suggests {goal_task}"
            confidence = goal_confidence
        elif data_task and goal_confidence < 0.5:
            # Low confidence in goal, use data analysis
            final_task = data_task.replace('binary_', '').replace('multiclass_', '').replace('ordinal_', '')
            reasoning = f"Goal analysis unclear, data suggests {data_task}"
            confidence = data_confidence
        elif data_task and data_task.startswith(goal_task):
            # Goal and data align
            final_task = goal_task
            reasoning = f"Goal analysis and data characteristics both suggest {goal_task}"
            confidence = min(goal_confidence + 0.2, 0.95)
        else:
            # Use goal analysis as primary, data as secondary
            final_task = goal_task
            reasoning = f"Using goal analysis ({goal_task}) as primary indication"
            confidence = goal_confidence
        
        # Generate algorithm recommendations
        algorithm_recommendations = self._recommend_algorithms(final_task, data_analysis)
        
        combined['recommendations'] = {
            'final_task_type': final_task,
            'confidence': confidence,
            'reasoning': reasoning,
            'algorithms': algorithm_recommendations,
            'learning_type': 'supervised' if final_task in ['classification', 'regression'] else 'unsupervised',
            'requires_target_column': final_task in ['classification', 'regression']
        }
        
        return combined

    def _recommend_algorithms(self, task_type: str, data_analysis: Dict) -> List[Dict[str, Any]]:
        """
        Recommend specific algorithms based on task type and data characteristics
        
        Args:
            task_type: ML task type
            data_analysis: Dataset analysis results
            
        Returns:
            List of algorithm recommendations with rationale
        """
        n_samples = data_analysis['shape'][0]
        n_features = data_analysis['n_features']
        has_categorical = data_analysis['feature_types']['categorical'] > 0
        has_missing = data_analysis['missing_data']['total_missing'] > 0
        
        algorithms = []
        
        if task_type == 'classification':
            # Random Forest - robust default choice
            algorithms.append({
                'name': 'Random Forest Classifier',
                'priority': 'high',
                'rationale': 'Handles mixed data types, robust to outliers, provides feature importance',
                'good_for': 'Most classification tasks, especially with mixed feature types'
            })
            
            # Gradient Boosting
            algorithms.append({
                'name': 'XGBoost/LightGBM',
                'priority': 'high',
                'rationale': 'Excellent performance, handles missing values, works well with structured data',
                'good_for': 'Structured data, competitive performance needed'
            })
            
            if n_samples < 1000:
                algorithms.append({
                    'name': 'Support Vector Machine',
                    'priority': 'medium',
                    'rationale': 'Works well with small datasets, especially with proper scaling',
                    'good_for': 'Small to medium datasets, high-dimensional data'
                })
            
            if not has_categorical and n_features < 50:
                algorithms.append({
                    'name': 'Logistic Regression',
                    'priority': 'medium',
                    'rationale': 'Simple, interpretable, fast training and prediction',
                    'good_for': 'Linear relationships, when interpretability is important'
                })
                
        elif task_type == 'regression':
            # Random Forest Regressor
            algorithms.append({
                'name': 'Random Forest Regressor', 
                'priority': 'high',
                'rationale': 'Handles mixed data types, captures non-linear relationships',
                'good_for': 'Most regression tasks, mixed feature types'
            })
            
            # Gradient Boosting
            algorithms.append({
                'name': 'XGBoost/LightGBM Regressor',
                'priority': 'high', 
                'rationale': 'State-of-the-art performance, handles missing values well',
                'good_for': 'Structured data, competitive performance needed'
            })
            
            if not has_categorical:
                algorithms.append({
                    'name': 'Linear Regression',
                    'priority': 'medium',
                    'rationale': 'Simple, fast, interpretable baseline',
                    'good_for': 'Linear relationships, when simplicity is preferred'
                })
                
                algorithms.append({
                    'name': 'Ridge/Lasso Regression',
                    'priority': 'medium',
                    'rationale': 'Regularized linear models, good for high-dimensional data',
                    'good_for': 'Many features, preventing overfitting'
                })
                
        elif task_type == 'clustering':
            algorithms.append({
                'name': 'K-Means',
                'priority': 'high',
                'rationale': 'Fast, simple, works well for spherical clusters',
                'good_for': 'Well-separated, spherical clusters, large datasets'
            })
            
            algorithms.append({
                'name': 'DBSCAN',
                'priority': 'medium',
                'rationale': 'Finds arbitrary-shaped clusters, handles outliers',
                'good_for': 'Non-spherical clusters, outlier detection needed'
            })
            
            if n_samples < 5000:
                algorithms.append({
                    'name': 'Hierarchical Clustering',
                    'priority': 'medium',
                    'rationale': 'No need to specify number of clusters, creates hierarchy',
                    'good_for': 'Small to medium datasets, when cluster hierarchy is important'
                })
                
        elif task_type == 'anomaly_detection':
            algorithms.append({
                'name': 'Isolation Forest',
                'priority': 'high',
                'rationale': 'Efficient for high-dimensional data, no assumption of data distribution',
                'good_for': 'High-dimensional data, numerical features'
            })
            
            algorithms.append({
                'name': 'One-Class SVM',
                'priority': 'medium',
                'rationale': 'Flexible decision boundary, works well with proper scaling',
                'good_for': 'Non-linear anomaly detection, small to medium datasets'
            })
            
        return algorithms

# Create global instance for easy import
goal_detector = GoalDetector()


def detect_ml_task(user_goal: str, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Main function to detect ML task type based on user goal and dataset
    
    Args:
        user_goal: User's description of their objective
        df: Input dataframe
        target_col: Target column name (optional)
        
    Returns:
        Complete analysis and recommendations
    """
    logger.info(f"Analyzing ML task for goal: '{user_goal[:100]}...'")
    
    # Analyze user goal
    goal_analysis = goal_detector.analyze_goal(user_goal)
    logger.info(f"Goal analysis suggests: {goal_analysis['primary_task']} (confidence: {goal_analysis['confidence']:.2f})")
    
    # Analyze dataset
    data_analysis = goal_detector.analyze_dataset_characteristics(df, target_col)
    logger.info(f"Dataset analysis complete: {data_analysis['shape'][0]} samples, {data_analysis['n_features']} features")
    
    # Combine analyses
    combined_analysis = goal_detector.combine_goal_and_data_analysis(goal_analysis, data_analysis)
    
    final_task = combined_analysis['recommendations']['final_task_type']
    confidence = combined_analysis['recommendations']['confidence']
    logger.info(f"Final recommendation: {final_task} (confidence: {confidence:.2f})")
    
    return combined_analysis