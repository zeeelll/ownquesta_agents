from typing import Any, Dict
import pandas as pd
from . import config
import logging

logger = logging.getLogger(__name__)


def summarize_df(file_path: str, n_head: int = None) -> Dict[str, Any]:
    """
    Generate comprehensive dataset summary using configurable parameters
    
    Args:
        file_path: Path to the CSV file
        n_head: Number of sample rows to include (uses config default if None)
        
    Returns:
        Dictionary containing dataset summary
    """
    try:
        logger.info(f"Starting EDA analysis for {file_path}")
        
        # Use configuration default if n_head not specified
        if n_head is None:
            n_head = config.DEFAULT_HEAD_COUNT
            
        # Load dataset
        df = pd.read_csv(file_path)
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Initialize summary dictionary
        summary: Dict[str, Any] = {}
        
        # Basic dataset info
        summary["dataset_info"] = {
            "shape": df.shape,
            "total_cells": df.shape[0] * df.shape[1],
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        }
        
        # Column information
        summary["columns"] = list(df.columns.astype(str))
        summary["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Data quality analysis
        missing_count = df.isnull().sum()
        total_rows = len(df)
        
        summary["data_quality"] = {
            "missing_count": missing_count.to_dict(),
            "missing_percentage": {col: round((count / total_rows) * 100, 2) 
                                 for col, count in missing_count.items()},
            "complete_rows": int(df.dropna().shape[0]),
            "completion_rate": round((df.dropna().shape[0] / total_rows) * 100, 2)
        }
        
        # Statistical summary using configurable percentiles
        try:
            desc = df.describe(
                include='all', 
                percentiles=config.DEFAULT_DESCRIBE_PERCENTILES
            ).to_dict()
            summary["statistical_summary"] = desc
            logger.debug("Statistical summary generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate statistical summary: {e}")
            summary["statistical_summary"] = {"error": f"Statistical summary failed: {str(e)}"}

        # Sample data
        try:
            sample = df.head(n_head).to_dict(orient="records")
            summary["sample_data"] = {
                "rows": sample,
                "sample_size": len(sample)
            }
        except Exception as e:
            logger.warning(f"Failed to generate sample data: {e}")
            summary["sample_data"] = {"error": f"Sample generation failed: {str(e)}"}

        # Column-wise cardinality for categorical analysis
        cardinality = {}
        categorical_candidates = []
        
        for col in df.columns:
            try:
                unique_count = int(df[col].nunique(dropna=True))
                cardinality[col] = unique_count
                
                # Identify potential categorical columns
                if unique_count <= config.MAX_CATEGORICAL_UNIQUE and unique_count > 1:
                    categorical_candidates.append({
                        "column": col,
                        "unique_values": unique_count,
                        "data_type": str(df[col].dtype)
                    })
            except Exception as e:
                logger.warning(f"Failed to analyze column {col}: {e}")
                cardinality[col] = -1
        
        summary["cardinality"] = cardinality
        summary["categorical_candidates"] = categorical_candidates
        
        # Data type analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        summary["column_types"] = {
            "numeric": {
                "columns": numeric_cols,
                "count": len(numeric_cols)
            },
            "categorical": {
                "columns": categorical_cols,
                "count": len(categorical_cols)
            },
            "datetime": {
                "columns": datetime_cols,
                "count": len(datetime_cols)
            }
        }
        
        # Potential target columns analysis
        potential_targets = []
        for col in df.columns:
            unique_vals = df[col].nunique(dropna=True)
            if 2 <= unique_vals <= config.MAX_CATEGORICAL_UNIQUE:
                potential_targets.append({
                    "column": col,
                    "unique_values": unique_vals,
                    "data_type": str(df[col].dtype),
                    "task_type": "classification"
                })
            elif unique_vals > config.MAX_CATEGORICAL_UNIQUE and col in numeric_cols:
                potential_targets.append({
                    "column": col,
                    "unique_values": unique_vals,
                    "data_type": str(df[col].dtype),
                    "task_type": "regression"
                })
        
        summary["potential_targets"] = potential_targets
        
        # Analysis metadata
        summary["analysis_info"] = {
            "sample_size": n_head,
            "max_categorical_unique": config.MAX_CATEGORICAL_UNIQUE,
            "percentiles_used": config.DEFAULT_DESCRIBE_PERCENTILES,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"EDA analysis completed successfully for {file_path}")
        return summary
        
    except Exception as e:
        error_msg = f"EDA analysis failed: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
