from typing import Dict
from . import config
import logging

logger = logging.getLogger(__name__)


def generate_docs(topic: str) -> Dict[str, str]:
    """
    Generate documentation and code examples for various ML topics
    
    Args:
        topic: The topic to generate documentation for
        
    Returns:
        Dictionary containing topic, explanation, and code
    """
    topic = (topic or "").strip().lower()
    logger.info(f"Generating documentation for topic: {topic}")
    
    if "eda" in topic or "explor" in topic or "analysis" in topic:
        text = (
            "Exploratory Data Analysis (EDA) helps you understand your data before modeling. "
            "Our comprehensive EDA includes dataset info, statistical summaries, missing value analysis, "
            "correlation detection, and actionable insights for data preparation."
        )
        code = '''# Comprehensive EDA Example - Easy to understand!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD YOUR DATA
df = pd.read_csv('your_dataset.csv')
print("✅ Data loaded successfully!")

# 2. BASIC DATASET INFORMATION
print("📊 DATASET OVERVIEW")
print("=" * 40)
print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Size: {df.size:,} total values")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# 3. COLUMN INFORMATION
print("\\n📝 COLUMN DETAILS")
print("=" * 40)
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns
print(f"Numeric columns ({len(numeric_cols)}): {list(numeric_cols)}")
print(f"Text columns ({len(categorical_cols)}): {list(categorical_cols)}")

# 4. MISSING VALUES CHECK
print("\\n🔍 MISSING VALUES ANALYSIS")
print("=" * 40)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

print("Missing values by column:")
for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.1f}%)")
    else:
        print(f"  {col}: ✅ No missing values")

# 5. STATISTICAL SUMMARY
print("\\n📈 STATISTICAL SUMMARY")
print("=" * 40)

# For numeric columns
if len(numeric_cols) > 0:
    print("\\nNumeric columns summary:")
    for col in numeric_cols[:3]:  # Show first 3 numeric columns
        series = df[col].dropna()
        print(f"\\n{col}:")
        print(f"  Mean: {series.mean():.2f}")
        print(f"  Median: {series.median():.2f}")
        print(f"  Std Dev: {series.std():.2f}")
        print(f"  Min-Max: {series.min():.2f} to {series.max():.2f}")
        print(f"  Skewness: {series.skew():.2f}")
        print(f"  Unique values: {series.nunique():,}")

# For text columns
if len(categorical_cols) > 0:
    print("\\nCategorical columns summary:")
    for col in categorical_cols[:3]:  # Show first 3 text columns
        series = df[col].dropna()
        print(f"\\n{col}:")
        print(f"  Unique values: {series.nunique()}")
        print(f"  Most common: {series.mode().iloc[0] if not series.mode().empty else 'N/A'}")
        print(f"  Top 3 values: {series.value_counts().head(3).to_dict()}")

# 6. DATA DISTRIBUTION ANALYSIS
print("\\n📊 OUTLIER DETECTION")
print("=" * 40)
for col in numeric_cols[:3]:  # Check first 3 numeric columns
    series = df[col].dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    outliers = series[(series < Q1 - 1.5*IQR) | (series > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(series)*100:.1f}%)")

# 7. CORRELATION ANALYSIS
print("\\n🔗 CORRELATION ANALYSIS")
print("=" * 40)
if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # High correlation
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        print("High correlations found:")
        for col1, col2, corr_val in high_corr_pairs:
            print(f"  {col1} ↔ {col2}: {corr_val:.3f}")
    else:
        print("No high correlations (>0.7) detected")

# 8. SIMPLE VISUALIZATIONS
print("\\n📈 CREATING VISUALIZATIONS...")
plt.figure(figsize=(15, 10))

# Missing values heatmap
plt.subplot(2, 3, 1)
sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
plt.title('Missing Values Pattern')

# Correlation heatmap
if len(numeric_cols) >= 2:
    plt.subplot(2, 3, 2)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')

# Distribution plots for first few numeric columns
for i, col in enumerate(numeric_cols[:4]):
    plt.subplot(2, 3, 3+i)
    df[col].hist(bins=20, alpha=0.7)
    plt.title(f'{col} Distribution')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 9. DATA QUALITY INSIGHTS
print("\\n💡 KEY INSIGHTS")
print("=" * 40)
total_missing_pct = (df.isnull().sum().sum() / df.size) * 100
print(f"Overall data completeness: {100-total_missing_pct:.1f}%")

constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
if constant_cols:
    print(f"⚠️  Constant columns (remove these): {constant_cols}")

duplicate_rows = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_rows} ({duplicate_rows/len(df)*100:.1f}%)")

print("\\n🎯 RECOMMENDATIONS:")
print("✅ Check missing values and decide on imputation strategy")
print("✅ Remove or investigate constant columns")
print("✅ Handle outliers if necessary")
print("✅ Consider encoding categorical variables for modeling")
'''
    
    elif "validate" in topic or "ml" in topic or "model" in topic:
        text = (
            "Model validation tests how well your machine learning models perform. "
            "Our system automatically handles data preprocessing, selects appropriate models, "
            "and provides comprehensive performance metrics with cross-validation."
        )
        code = '''# Machine Learning Validation - Step by step!

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. LOAD AND PREPARE DATA
print("🚀 STARTING ML VALIDATION")
print("=" * 50)

df = pd.read_csv('your_dataset.csv')
target_column = 'your_target_column'  # Replace with your target

print(f"Original dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 2. HANDLE MISSING VALUES IN TARGET
print("\\n🎯 TARGET ANALYSIS")
print("=" * 30)
target_missing = df[target_column].isnull().sum()
print(f"Missing values in target: {target_missing}")

# Remove rows where target is missing
df_clean = df.dropna(subset=[target_column]).copy()
print(f"Clean dataset: {df_clean.shape[0]:,} rows")

# Separate features and target
X = df_clean.drop(columns=[target_column])
y = df_clean[target_column]

# 3. DETERMINE TASK TYPE
unique_targets = y.nunique()
task_type = "classification" if unique_targets <= 20 else "regression"
print(f"\\nTask type: {task_type.upper()}")
print(f"Target has {unique_targets} unique values")

if task_type == "classification":
    print(f"Class distribution:")
    print(y.value_counts())

# 4. SMART DATA PREPROCESSING
print("\\n🔧 DATA PREPROCESSING")
print("=" * 30)

# Identify column types
numeric_cols = X.select_dtypes(include=['number']).columns
text_cols = X.select_dtypes(include=['object']).columns

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Text columns: {len(text_cols)}")

# Handle missing values
print("\\nHandling missing values...")

# For numeric columns - use median
if len(numeric_cols) > 0:
    numeric_imputer = SimpleImputer(strategy='median')
    X[numeric_cols] = numeric_imputer.fit_transform(X[numeric_cols])
    print(f"✅ Fixed missing values in {len(numeric_cols)} numeric columns")

# For text columns - use most frequent value
if len(text_cols) > 0:
    for col in text_cols:
        mode_value = X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown"
        X[col] = X[col].fillna(mode_value)
    print(f"✅ Fixed missing values in {len(text_cols)} text columns")

# Encode categorical variables (text to numbers)
print("\\nEncoding categorical variables...")
for col in text_cols:
    if X[col].nunique() <= 20:  # Low cardinality
        # One-hot encoding (create dummy variables)
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
        print(f"✅ One-hot encoded: {col}")
    else:  # High cardinality
        # Label encoding (assign numbers)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        print(f"✅ Label encoded: {col}")

# Scale numeric features (important for many algorithms)
current_numeric_cols = X.select_dtypes(include=['number']).columns
if len(current_numeric_cols) > 0:
    scaler = RobustScaler()  # Works well with outliers
    X[current_numeric_cols] = scaler.fit_transform(X[current_numeric_cols])
    print(f"✅ Scaled {len(current_numeric_cols)} numeric features")

print(f"\\nFinal feature matrix: {X.shape[0]:,} rows × {X.shape[1]} columns")

# 5. SPLIT DATA FOR TRAINING AND TESTING
print("\\n✂️  SPLITTING DATA")
print("=" * 25)

if task_type == "classification":
    # Encode target for classification
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y.astype(str))
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.2,      # 20% for testing
        random_state=42,    # For reproducible results
        stratify=y_encoded  # Keep class proportions
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set: {X_train.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# 6. TRAIN AND EVALUATE MODELS
print("\\n🤖 MODEL TRAINING & EVALUATION")
print("=" * 40)

if task_type == "classification":
    # Try multiple classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = 0
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        print("\\nDetailed Performance:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix (shows correct vs wrong predictions)
        cm = confusion_matrix(y_test, y_pred)
        print(f"\\nConfusion Matrix:")
        print(cm)
        
        # Track best model
        if accuracy > best_score:
            best_score = accuracy
            best_model = name
            
        print("-" * 50)
    
    print(f"🏆 BEST MODEL: {best_model} with {best_score*100:.1f}% accuracy")
    
else:  # Regression
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        print(f"\\nTraining {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate regression metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        
        print(f"R² Score: {r2:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = name
            
        print("-" * 50)
    
    print(f"🏆 BEST MODEL: {best_model} with R² = {best_score:.4f}")

# 7. FINAL INSIGHTS
print("\\n💡 KEY INSIGHTS & RECOMMENDATIONS")
print("=" * 45)
print("✅ Data preprocessing completed successfully")
print("✅ Multiple models trained and compared")
print("✅ Best performing model identified")
print("\\n🎯 Next Steps:")
print("1. Try feature engineering (create new features)")
print("2. Experiment with hyperparameter tuning")
print("3. Collect more data if performance is low")
print("4. Consider ensemble methods for better results")
'''

    elif "config" in topic or "setup" in topic or "environment" in topic:
        text = (
            "Proper configuration is key to reliable ML workflows. The ML Assistant uses "
            "environment variables for flexible configuration. Key settings include file size limits, "
            "model parameters, and processing thresholds."
        )
        code = f'''# Environment Configuration (.env file)

# ML Assistant specific settings
ML_ASSISTANT_MAX_FILE_SIZE_MB={config.MAX_FILE_SIZE_MB}
ML_ASSISTANT_ALLOWED_EXTENSIONS={",".join(config.ALLOWED_EXTENSIONS)}
ML_ASSISTANT_DEFAULT_TEST_SIZE={config.DEFAULT_TEST_SIZE}
ML_ASSISTANT_RANDOM_STATE={config.DEFAULT_RANDOM_STATE}
ML_ASSISTANT_MAX_CATEGORICAL_UNIQUE={config.MAX_CATEGORICAL_UNIQUE}

# Model training settings
ML_ASSISTANT_LOGISTIC_MAX_ITER={config.LOGISTIC_REGRESSION_MAX_ITER}
ML_ASSISTANT_MODEL_TIMEOUT={config.MODEL_TIMEOUT_SECONDS}

# Data quality thresholds
ML_ASSISTANT_MIN_SAMPLES={config.MIN_SAMPLES_FOR_VALIDATION}
ML_ASSISTANT_MIN_FEATURES={config.MIN_FEATURES_FOR_VALIDATION}

# OpenAI settings
ML_ASSISTANT_CHAT_MODEL={config.OPENAI_CHAT_MODEL}
ML_ASSISTANT_TEMPERATURE={config.OPENAI_TEMPERATURE}
ML_ASSISTANT_MAX_TOKENS={config.OPENAI_MAX_TOKENS}

# Usage in Python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Access settings
max_file_size = int(os.getenv("ML_ASSISTANT_MAX_FILE_SIZE_MB", "100"))
test_size = float(os.getenv("ML_ASSISTANT_DEFAULT_TEST_SIZE", "0.2"))

print(f"Configuration loaded:")
print(f"Max file size: {{max_file_size}}MB")
print(f"Test size: {{test_size}}")
'''

    elif "api" in topic or "endpoint" in topic or "integration" in topic:
        text = (
            "The ML Assistant API makes machine learning accessible through simple REST endpoints. "
            "Upload your data, get comprehensive analysis, train models, and chat with AI - all through "
            "easy-to-use web requests with detailed responses and error handling."
        )
        code = '''# Complete ML Assistant API Guide - Easy Examples!

import requests
import json

# 🌐 API Setup
base_url = "http://localhost:8000/ml-assistant"
print("🚀 ML Assistant API Integration Guide")
print("=" * 50)

# 1️⃣ CHECK API HEALTH
print("\\n1. Health Check")
print("-" * 20)

try:
    response = requests.get(f"{base_url}/health")
    health_data = response.json()
    
    print(f"✅ API Status: {health_data['status']}")
    print(f"📊 OpenAI Available: {health_data['openai_status']}")
    print(f"💾 Max file size: {health_data['max_file_size_mb']}MB")
    print(f"📁 Allowed files: {', '.join(health_data['allowed_extensions'])}")
    
except Exception as e:
    print(f"❌ API not accessible: {e}")
    print("💡 Make sure the server is running on localhost:8000")

# 2️⃣ UPLOAD YOUR DATA
print("\\n\\n2. Upload Dataset")
print("-" * 25)

# Replace 'your_dataset.csv' with your actual file
dataset_file = "your_dataset.csv"

try:
    with open(dataset_file, "rb") as f:
        files = {"file": f}
        response = requests.post(f"{base_url}/upload", files=files)
    
    if response.status_code == 200:
        upload_result = response.json()
        print(f"✅ File uploaded successfully!")
        print(f"📁 Filename: {upload_result['filename']}")
        print(f"📊 Size: {upload_result['size_mb']} MB")
        print(f"⏰ Upload time: {upload_result['upload_time']}")
        
        # Show basic dataset info
        summary = upload_result['summary']
        dataset_info = summary['dataset_info']
        print(f"\\n📈 Dataset Overview:")
        print(f"   Shape: {dataset_info['rows']:,} rows × {dataset_info['columns']} columns")
        print(f"   Memory: {dataset_info['memory_usage_mb']:.1f} MB")
        
        filename = upload_result['filename']  # Save for next steps
    else:
        print(f"❌ Upload failed: {response.json()}")
        
except FileNotFoundError:
    print(f"❌ File '{dataset_file}' not found")
    print("💡 Place your CSV file in the same directory as this script")
    filename = "demo_dataset.csv"  # Use demo name for examples

# 3️⃣ COMPREHENSIVE EDA
print("\\n\\n3. Comprehensive Analysis (EDA)")
print("-" * 40)

eda_data = {"filename": filename}
response = requests.post(f"{base_url}/eda", data=eda_data)

if response.status_code == 200:
    eda_result = response.json()
    print(f"✅ EDA completed in {eda_result['processing_time']} seconds")
    
    # Show key insights from EDA
    eda_details = eda_result['eda']
    
    # Dataset info
    if 'dataset_info' in eda_details:
        info = eda_details['dataset_info']
        print(f"\\n📊 Dataset Details:")
        print(f"   Total cells: {info['size']:,}")
        print(f"   Memory usage: {info['memory_usage_mb']:.1f} MB")
    
    # Missing values
    if 'missing_values' in eda_details:
        missing = eda_details['missing_values']
        print(f"\\n🔍 Data Quality:")
        print(f"   Missing data: {missing['missing_percentage_total']:.1f}%")
        print(f"   Columns with missing values: {missing['columns_with_missing']}")
    
    # Show insights
    if 'insights' in eda_details:
        print(f"\\n💡 Key Insights:")
        for insight in eda_details['insights'][:5]:  # Show top 5 insights
            print(f"   • {insight}")
else:
    print(f"❌ EDA failed: {response.json()}")

# 4️⃣ MODEL VALIDATION
print("\\n\\n4. Model Training & Validation")
print("-" * 40)

# You need to specify your target column here
target_column = "your_target_column"  # Replace with your actual target

validate_data = {
    "filename": filename,
    "target": target_column
}
response = requests.post(f"{base_url}/validate", data=validate_data)

if response.status_code == 200:
    validation_result = response.json()
    print(f"✅ Validation completed in {validation_result['processing_time']} seconds")
    
    # Show validation summary
    result = validation_result['result']
    
    print(f"\\n🎯 Task: {result['task_type'].upper()}")
    
    # Data summary
    if 'data_summary' in result:
        data_info = result['data_summary']
        print(f"\\n📊 Data Used:")
        print(f"   Samples: {data_info['samples_used']:,}")
        print(f"   Features: {data_info['features_processed']}")
        print(f"   Target values: {data_info['target_unique_values']}")
    
    # Model performance
    if 'models' in result:
        models = result['models']
        print(f"\\n🤖 Model Performance:")
        
        for model_name, model_results in models.items():
            if 'error' not in model_results:
                print(f"\\n   {model_name.replace('_', ' ').title()}:")
                if result['task_type'] == 'classification':
                    print(f"      Accuracy: {model_results['accuracy']*100:.1f}%")
                    print(f"      F1-Score: {model_results['f1_score']:.3f}")
                else:
                    print(f"      R² Score: {model_results['r2_score']:.3f}")
                    print(f"      RMSE: {model_results['root_mean_squared_error']:.3f}")
        
        if 'best_model' in result:
            print(f"\\n🏆 Best Model: {result['best_model'].replace('_', ' ').title()}")
            
else:
    print(f"❌ Validation failed: {response.json()}")
    print("💡 Make sure your target column name is correct")

# 5️⃣ AI CHAT ASSISTANCE
print("\\n\\n5. AI Chat Assistant")
print("-" * 30)

chat_questions = [
    "How can I improve my model accuracy?",
    "What should I do about missing values?",
    "How do I handle categorical variables?"
]

for question in chat_questions[:1]:  # Ask first question
    chat_data = {
        "message": question,
        "context": f"Working with a {result.get('task_type', 'machine learning')} problem"
    }
    
    response = requests.post(f"{base_url}/chat", json=chat_data)
    
    if response.status_code == 200:
        chat_result = response.json()
        print(f"\\n❓ Question: {question}")
        print(f"🤖 AI Response: {chat_result['reply'][:200]}...")
        print(f"⚡ Model used: {chat_result['model']}")
        break
    else:
        print(f"❌ Chat failed: {response.json()}")

# 6️⃣ GET DOCUMENTATION
print("\\n\\n6. Get Help & Documentation")
print("-" * 40)

docs_topics = ["eda", "validation", "api"]
docs_data = {"topic": "eda"}  # Get EDA documentation

response = requests.post(f"{base_url}/docs", data=docs_data)

if response.status_code == 200:
    docs_result = response.json()
    print(f"✅ Documentation for: {docs_result['topic']}")
    print(f"📝 Explanation: {docs_result['explanation'][:150]}...")
    print(f"💻 Code example available: {len(docs_result['code'])} characters")
else:
    print(f"❌ Docs failed: {response.json()}")

# 7️⃣ LIST UPLOADED FILES
print("\\n\\n7. Manage Your Files")
print("-" * 30)

response = requests.get(f"{base_url}/files")

if response.status_code == 200:
    files_result = response.json()
    print(f"✅ Files in upload directory:")
    
    for file_info in files_result['files']:
        print(f"   📁 {file_info['filename']} ({file_info['size_mb']} MB)")
        print(f"      Modified: {file_info['modified']}")
else:
    print(f"❌ File list failed: {response.json()}")

# 🎯 SUMMARY
print("\\n\\n🎯 COMPLETE WORKFLOW SUMMARY")
print("=" * 50)
print("✅ 1. Upload your CSV dataset")
print("✅ 2. Get comprehensive EDA analysis") 
print("✅ 3. Train and validate ML models")
print("✅ 4. Chat with AI for guidance")
print("✅ 5. Access documentation and examples")
print("✅ 6. Manage your uploaded files")
print("\\n💡 The API handles all the complex ML work for you!")
print("🚀 Just upload data and get professional results!")
'''

    else:
        text = (
            "The ML Assistant provides comprehensive support for machine learning workflows. "
            "Available topics include: EDA (exploratory data analysis), model validation, "
            "API integration, and configuration setup. Each component is designed with "
            "best practices, proper error handling, and configurable parameters."
        )
        code = '''# Available documentation topics:

# 1. EDA and Data Analysis
# - Exploratory data analysis workflows
# - Data quality assessment
# - Statistical summaries and visualizations

# 2. Model Validation
# - Automated task type detection
# - Training and evaluation pipelines
# - Performance metrics and reporting

# 3. API Integration
# - REST API endpoints
# - File upload and processing
# - Real-time AI assistance

# 4. Configuration
# - Environment variable setup
# - Performance tuning
# - Security settings

# Request specific documentation:
# POST /docs with topic parameter:
# - "eda" or "analysis" for data analysis
# - "validate" or "model" for ML validation
# - "api" or "integration" for API usage
# - "config" or "setup" for configuration

# Example API call:
import requests

response = requests.post(
    "http://localhost:8000/ml-assistant/docs",
    data={"topic": "your_topic_here"}
)
print(response.json())
'''

    result = {
        "topic": topic if topic else "general",
        "explanation": text,
        "code": code
    }
    
    logger.info(f"Documentation generated for topic: {topic}")
    return result
