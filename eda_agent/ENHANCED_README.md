# Enhanced EDA Agent - Comprehensive Data Analysis

The Enhanced EDA (Exploratory Data Analysis) Agent provides comprehensive statistical analysis and data insights for CSV datasets. It performs advanced analytics including distribution analysis, correlation studies, outlier detection, and data quality assessment.

## Features

### 🔍 Dataset Overview
- **Shape Analysis**: Rows, columns, memory usage, file size
- **Data Quality**: Missing values percentage, duplicate detection
- **Column Types**: Automatic classification (numeric, categorical, datetime, boolean)
- **Memory Optimization**: Detailed memory usage analysis

### 📊 Comprehensive Statistics

#### Numeric Columns
- **Central Tendency**: Mean, median, mode
- **Variability**: Standard deviation, variance, range, IQR
- **Distribution Shape**: Skewness, kurtosis, coefficient of variation
- **Percentiles**: 5th, 10th, 25th, 50th, 75th, 90th, 95th, 99th
- **Advanced Metrics**: Outlier counts and percentages

#### Categorical Columns  
- **Diversity Metrics**: Unique count, entropy, top values
- **Frequency Analysis**: Most common values, distribution patterns
- **Data Quality**: Rare values detection, common patterns

### 📈 Advanced Distribution Analysis
- **Normality Testing**: Shapiro-Wilk, Kolmogorov-Smirnov, D'Agostino tests
- **Distribution Shape**: Detailed skewness and kurtosis interpretation
- **Distribution Identification**: Automatic pattern recognition
- **Transformation Recommendations**: Data normalization suggestions
- **Chart Data**: Histogram bins, box plot data, density plot preparation

### 🔗 Advanced Correlation Analysis
- **Multiple Methods**: Pearson, Spearman, and Kendall correlations
- **Strong Correlations**: Automatic detection of relationships (|r| > 0.7)
- **Moderate Correlations**: Medium strength relationships (|r| > 0.5)
- **Multicollinearity**: VIF approximation and risk assessment
- **Visual Data**: Correlation heatmap preparation

### 🎯 Outlier Detection
- **Multiple Methods**: 
  - IQR Method (Interquartile Range)
  - Z-Score Method
  - Modified Z-Score (Median Absolute Deviation)
  - Isolation Forest Approximation
- **Consensus Outliers**: Multi-method agreement
- **Impact Analysis**: Statistical effect of outlier removal
- **Visualization Data**: Scatter plots, box plots for outliers

### 🔍 Data Quality Analysis
- **Missing Values**: Detailed per-column analysis
- **Duplicate Detection**: Full and subset duplicate identification
- **Data Consistency**: Mixed data types, casing issues, potential date columns
- **Memory Analysis**: Per-column and total memory usage
- **Recommendations**: Actionable data cleaning suggestions

### 📊 Chart Data Preparation
All analysis includes prepared data for visualization:
- **Histograms**: Bin counts, centers, and edges
- **Box Plots**: Quartiles, whiskers, outliers
- **Correlation Heatmaps**: Matrix data for visualization
- **Scatter Plots**: Outlier highlighting data

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

### Command Line Usage
```bash
# Run comprehensive analysis
python config.py dataset.csv

# Dry run (no LLM, tools only)
python config.py dataset.csv --dry-run

# Custom temperature
python config.py dataset.csv --temperature 0.2
```

### Python API
```python
from eda_agent import create_agent

agent = create_agent()
result = agent.run("Perform complete exploratory data analysis on file data.csv")
print(result)
```

### FastAPI Endpoint
```bash
# Start the server (from main project)
# POST /eda/upload_and_run
curl -X POST "http://localhost:8002/eda/upload_and_run" -F "file=@dataset.csv"
```

## API Response Structure

```json
{
  "filename": "data.csv",
  "status": "done",
  "results": {
    "dataset_overview": {
      "shape": {"rows": 1000, "columns": 15},
      "size": {"memory_usage_mb": 12.5, "file_size_mb": 8.2},
      "data_quality": {"missing_percentage": 3.2, "duplicate_rows": 5},
      "column_types": {"numeric": 10, "categorical": 5}
    },
    "comprehensive_statistics": {
      "numeric_statistics": {
        "column_name": {
          "mean": 45.6, "median": 42.1, "std": 12.3,
          "skewness": 0.15, "kurtosis": -0.8,
          "outliers": {"outlier_count": 3, "percentage": 0.3}
        }
      },
      "categorical_statistics": {...}
    },
    "advanced_distribution_analysis": {
      "column_name": {
        "normality_tests": {
          "shapiro_wilk": {"p_value": 0.023, "is_normal": false},
          "kolmogorov_smirnov": {"p_value": 0.034, "is_normal": false}
        },
        "distribution_shape": {
          "skewness": 1.23, "skewness_interpretation": "moderately right skewed"
        },
        "conclusion": {
          "likely_normal": false,
          "recommendation": "Consider log transformation to reduce right skew"
        }
      }
    },
    "advanced_correlation_analysis": {
      "correlation_insights": {
        "strong_correlations": [
          {"variable_1": "height", "variable_2": "weight", "correlation": 0.87}
        ]
      },
      "multicollinearity_analysis": {...}
    },
    "data_quality_analysis": {
      "missing_values": {...},
      "duplicate_analysis": {...},
      "recommendations": ["Consider dropping columns with >50% missing values"]
    },
    "outlier_detection": {
      "summary": {
        "total_outliers": 25,
        "overall_outlier_percentage": 2.5
      },
      "outlier_analysis": {...}
    }
  }
}
```

## Integration with ML Validation

The EDA agent integrates with the machine learning validation page at `/validate`:

1. **Upload Dataset**: Select CSV file for analysis
2. **Automatic Validation**: Data quality checks
3. **Comprehensive EDA**: Advanced statistical analysis
4. **Visual Dashboard**: Rich, interactive display
5. **ML Readiness**: Preprocessing recommendations

## Available Tools

| Tool | Description |
|------|-------------|
| `dataset_overview` | Complete dataset summary with memory usage |
| `column_analysis` | Detailed per-column analysis |
| `comprehensive_statistics` | Full statistical measures for all column types |
| `advanced_distribution_analysis` | Distribution testing and normality assessment |
| `advanced_correlation_analysis` | Multi-method correlation with multicollinearity |
| `data_quality_analysis` | Missing values, duplicates, consistency issues |
| `outlier_detection` | Multi-method outlier detection with consensus |
| `dataset_shape` | Basic shape information |
| `column_names` | List of column names |
| `statistical_measures` | Basic statistical measures |
| `correlation_matrix` | Simple correlation matrix |

## Output Interpretation

### Statistical Measures
- **Mean vs Median**: If difference > 10% of std dev, data may be skewed
- **Coefficient of Variation**: >30% indicates high variability
- **Skewness**: |skew| > 1 indicates significant skewness
- **Kurtosis**: |kurt| > 1 indicates heavy/light tails

### Normality Assessment
- **Multiple Tests**: Use consensus of 2+ tests for reliability
- **p-value > 0.05**: Data likely normal (fail to reject H0)
- **p-value ≤ 0.05**: Data significantly non-normal

### Correlation Interpretation
- **|r| > 0.7**: Strong linear relationship
- **0.5 < |r| ≤ 0.7**: Moderate relationship
- **|r| ≤ 0.3**: Weak relationship
- **VIF > 10**: High multicollinearity risk

### Data Quality Flags
- **Missing > 50%**: Consider column removal
- **Duplicates > 5%**: Investigate data collection process
- **Mixed data types**: Data cleaning required
- **Inconsistent casing**: Standardization needed

## Dependencies

```
scipy>=1.11.0          # Statistical functions
pandas>=2.0.0          # Data manipulation
numpy>=1.24.0          # Numerical computing
langchain>=0.1.0       # LLM integration
fastapi>=0.100.0       # Web API
python-dotenv>=1.0.0   # Environment management
```

## Environment Configuration

```bash
# Optional: For LLM-powered insights
OPENAI_API_KEY=your_openai_api_key
```

## Performance Considerations

- **Large Datasets**: Automatic sampling for expensive operations (>5000 rows)
- **Memory Optimization**: Efficient data type handling
- **Error Handling**: Graceful degradation for edge cases
- **Timeout Protection**: Safe processing of large files

This enhanced EDA agent provides enterprise-grade data analysis capabilities with comprehensive statistical insights, making it ideal for data science workflows and machine learning preprocessing.