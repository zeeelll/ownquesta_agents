#!/usr/bin/env python3
"""
Test script for the enhanced EDA agent.
This script demonstrates all the new features and generates a sample dataset for testing.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def create_sample_dataset():
    """Create a sample dataset with various data patterns for testing all EDA features."""
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample data with different characteristics
    data = {
        # Numeric columns with different distributions
        'normal_data': np.random.normal(50, 10, n_samples),
        'skewed_data': np.random.exponential(2, n_samples),
        'uniform_data': np.random.uniform(0, 100, n_samples),
        'correlated_data': None,  # Will be created based on normal_data
        
        # Categorical columns
        'category_A': np.random.choice(['Type1', 'Type2', 'Type3', 'Type4'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'category_B': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        
        # Column with missing values
        'missing_data': np.random.normal(25, 5, n_samples),
        
        # Column with outliers
        'outlier_data': np.random.normal(100, 15, n_samples),
        
        # Mixed data types (problematic column)
        'mixed_data': ['text_' + str(i) if i % 10 == 0 else str(i) for i in range(n_samples)],
        
        # Date-like strings
        'date_strings': [f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n_samples)]
    }
    
    # Create correlated data
    data['correlated_data'] = data['normal_data'] * 0.8 + np.random.normal(0, 5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    df.loc[missing_indices, 'missing_data'] = np.nan
    
    # Add some extreme outliers
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[outlier_indices, 'outlier_data'] = np.random.choice([300, -50, 500, -100], size=20)
    
    # Add duplicate rows
    duplicate_rows = df.sample(n=50, random_state=42)
    df = pd.concat([df, duplicate_rows], ignore_index=True)
    
    return df

def test_enhanced_eda():
    """Test the enhanced EDA functionality."""
    # Create sample dataset
    print("Creating sample dataset...")
    df = create_sample_dataset()
    
    # Save to CSV
    test_file = Path(__file__).parent / "test_data.csv"
    df.to_csv(test_file, index=False)
    print(f"Sample dataset created: {test_file}")
    print(f"Dataset shape: {df.shape}")
    
    # Import EDA tools
    try:
        from config import (
            dataset_overview,
            comprehensive_statistics,
            advanced_distribution_analysis,
            advanced_correlation_analysis,
            data_quality_analysis,
            outlier_detection
        )
        print("Successfully imported enhanced EDA tools!")
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Test each enhanced tool
    tools_to_test = [
        ("Dataset Overview", dataset_overview),
        ("Comprehensive Statistics", comprehensive_statistics),
        ("Advanced Distribution Analysis", advanced_distribution_analysis),
        ("Advanced Correlation Analysis", advanced_correlation_analysis),
        ("Data Quality Analysis", data_quality_analysis),
        ("Outlier Detection", outlier_detection),
    ]
    
    print("\n" + "="*50)
    print("TESTING ENHANCED EDA TOOLS")
    print("="*50)
    
    for tool_name, tool_func in tools_to_test:
        print(f"\nTesting {tool_name}...")
        try:
            result = tool_func(str(test_file))
            print(f"✅ {tool_name}: SUCCESS")
            # Show first 200 characters of result
            result_preview = result[:200] + "..." if len(result) > 200 else result
            print(f"Preview: {result_preview}")
        except Exception as e:
            print(f"❌ {tool_name}: FAILED - {e}")
    
    print(f"\n🧹 Cleaning up test file: {test_file}")
    test_file.unlink(missing_ok=True)
    
    print("\n" + "="*50)
    print("ENHANCED EDA TESTING COMPLETED!")
    print("="*50)

if __name__ == "__main__":
    test_enhanced_eda()