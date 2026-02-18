import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, Code, BarChart3, CheckCircle, XCircle, Loader2, FileText, TrendingUp, AlertCircle, Target, Brain, Play, Database, Zap, MessageSquare } from 'lucide-react';

const ValidationAgenticAI = () => {
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, results
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [edaResults, setEdaResults] = useState(null);
  const [mlResults, setMlResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [detectedGoal, setDetectedGoal] = useState(null);
  const [showCode, setShowCode] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [questions, setQuestions] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [mlValidationResult, setMlValidationResult] = useState(null);
  const fileInputRef = useRef(null);

  const detectGoal = (userInput) => {
    const input = userInput.toLowerCase();

    // Supervised learning keywords
    const supervisedKeywords = ['predict', 'classification', 'regression', 'forecast', 'supervised',
                                'target', 'label', 'outcome', 'predict price', 'predict sales',
                                'classify', 'categorize', 'model', 'train'];

    // Unsupervised learning keywords
    const unsupervisedKeywords = ['cluster', 'segment', 'pattern', 'group', 'unsupervised',
                                  'anomaly', 'outlier', 'similarity', 'discover', 'grouping'];

    // EDA/Analysis keywords
    const edaKeywords = ['analyze', 'explore', 'understand', 'insights', 'statistics',
                         'visualize', 'eda', 'exploratory', 'summary'];

    const supervisedScore = supervisedKeywords.filter(kw => input.includes(kw)).length;
    const unsupervisedScore = unsupervisedKeywords.filter(kw => input.includes(kw)).length;
    const edaScore = edaKeywords.filter(kw => input.includes(kw)).length;

    if (supervisedScore > unsupervisedScore && supervisedScore > edaScore) {
      return {
        type: 'supervised',
        description: 'Supervised Learning (Prediction/Classification)',
        focus: ['Target variable identification', 'Feature importance', 'Missing value impact',
                'Class balance (if classification)', 'Feature correlations with target']
      };
    } else if (unsupervisedScore > supervisedScore && unsupervisedScore > edaScore) {
      return {
        type: 'unsupervised',
        description: 'Unsupervised Learning (Clustering/Pattern Discovery)',
        focus: ['Feature scaling requirements', 'Outlier detection', 'Feature variance',
                'Correlation patterns', 'Dimensionality considerations']
      };
    } else {
      return {
        type: 'eda',
        description: 'Exploratory Data Analysis',
        focus: ['Data quality assessment', 'Statistical summaries', 'Distribution analysis',
                'Correlation insights', 'Missing data patterns']
      };
    }
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    if (!uploadedFile.name.endsWith('.csv')) {
      alert('Please upload a CSV file only.');
      return;
    }

    setFile(uploadedFile);

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const csvText = event.target.result;
        setDataset(csvText);

        // Quick parse for display
        const lines = csvText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = lines.slice(1);

        setCurrentStep('goal_input');
      } catch (error) {
        alert('Error parsing CSV file. Please ensure it\'s properly formatted.');
      }
    };
    reader.readAsText(uploadedFile);
  };

  const startAnalysis = async () => {
    if (!input.trim() || !dataset) return;

    const goal = detectGoal(input);
    setDetectedGoal(goal);
    setCurrentStep('processing');
    setIsProcessing(true);

    try {
      // Call ML validation endpoint which includes EDA
      const validationUrl = (typeof window !== 'undefined' && (window.__OWNQUESTA_VALIDATION_URL || window.OWNQUESTA_VALIDATION_URL)) || 'http://localhost:8000/validation/validate';
      const response = await fetch(validationUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csv_text: dataset,
          goal: goal
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === 'success') {
        setEdaResults(result.eda_result);
        if (result.ml_result) {
          setMlResults(result.ml_result);
          setMlValidationResult(result.ml_result);
        } else {
          setMlResults(result);
          setMlValidationResult(result);
        }
        setCurrentStep('results');
      } else {
        throw new Error('Analysis failed');
      }
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Analysis failed: ${error.message}`);
      setCurrentStep('goal_input');
    } finally {
      setIsProcessing(false);
    }
  };

  const generateCode = () => {
    if (!edaResults) return 'No EDA results available';
    
    return `# Generated EDA and ML Pipeline Code
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv('your_dataset.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# EXPLORATORY DATA ANALYSIS
print("\n=== DATASET OVERVIEW ===")
print("Dataset Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes)

print("\n=== MISSING VALUES ===")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

print("\n=== STATISTICAL SUMMARY ===")
print(df.describe())

# VISUALIZATION SECTION
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Dataset Overview', fontsize=16)

# Missing values heatmap
if df.isnull().sum().sum() > 0:
    axes[0,0].title.set_text('Missing Values Heatmap')
    sns.heatmap(df.isnull(), cbar=True, ax=axes[0,0])
else:
    axes[0,0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
    axes[0,0].set_title('Missing Values Check')

# Correlation matrix
numerical_cols = df.select_dtypes(include=[np.number]).columns
if len(numerical_cols) > 1:
    corr_matrix = df[numerical_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
    axes[0,1].set_title('Feature Correlation Matrix')
else:
    axes[0,1].text(0.5, 0.5, 'Insufficient numerical features', ha='center', va='center')

# Data types distribution
data_types = df.dtypes.astype(str).value_counts()
axes[1,0].pie(data_types.values, labels=data_types.index, autopct='%1.1f%%')
axes[1,0].set_title('Data Types Distribution')

# Dataset size breakdown
axes[1,1].bar(['Rows', 'Columns', 'Numeric', 'Categorical'], 
              [df.shape[0], df.shape[1], len(numerical_cols), len(df.select_dtypes(include=['object']).columns)])
axes[1,1].set_title('Dataset Composition')

plt.tight_layout()
plt.show()

# FEATURE ANALYSIS
print("\n=== FEATURE ANALYSIS ===")

# Numerical features
if len(numerical_cols) > 0:
    print(f"\nNumerical features ({len(numerical_cols)}): {list(numerical_cols)}")
    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std: {df[col].std():.2f}")
        print(f"  Skewness: {df[col].skew():.2f}")
        print(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")

# Categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    print(f"\nCategorical features ({len(categorical_cols)}): {list(categorical_cols)}")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"  Values: {df[col].value_counts().head().to_dict()}")
        print(f"  Missing: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")

print("\n=== ANALYSIS COMPLETE ===")
print("Next steps: Choose target variable and run ML models")
`;
  };
    if (!currentQuestion.trim() || !edaResults) return;

    const question = currentQuestion.trim();
    setCurrentQuestion('');

    try {
      const response = await fetch('/validation/question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          eda_results: edaResults
        })
      });

      if (response.ok) {
        const result = await response.json();
        setQuestions(prev => [...prev, {
          question: question,
          answer: result.answer,
          timestamp: new Date().toLocaleTimeString()
        }]);
      } else {
        setQuestions(prev => [...prev, {
          question: question,
          answer: 'Sorry, I couldn\'t process your question. Please try rephrasing.',
          timestamp: new Date().toLocaleTimeString()
        }]);
      }
    } catch (error) {
      setQuestions(prev => [...prev, {
        question: question,
        answer: 'Error connecting to analysis service. Please try again.',
        timestamp: new Date().toLocaleTimeString()
      }]);
    }
  };

  const renderOverview = () => {
    if (!edaResults) return null;

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-blue-600" />
            Dataset Overview
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded">
              <div className="text-2xl font-bold text-blue-600">{edaResults.shape.rows}</div>
              <div className="text-sm text-gray-600">Rows</div>
            </div>
            <div className="text-center p-4 bg-green-50 rounded">
              <div className="text-2xl font-bold text-green-600">{edaResults.shape.columns}</div>
              <div className="text-sm text-gray-600">Columns</div>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded">
              <div className="text-2xl font-bold text-purple-600">{edaResults.numericColumns?.length || 0}</div>
              <div className="text-sm text-gray-600">Numeric</div>
            </div>
            <div className="text-center p-4 bg-orange-50 rounded">
              <div className="text-2xl font-bold text-orange-600">{edaResults.objectColumns?.length || 0}</div>
              <div className="text-sm text-gray-600">Categorical</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-green-600" />
            Analysis Goal
          </h3>
          <div className="p-4 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-200">
            <div className="font-medium text-lg text-gray-800 mb-2">{detectedGoal?.description}</div>
            <div className="text-sm text-gray-600">Focus Areas:</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
              {detectedGoal?.focus.slice(0, 4).map((item, index) => (
                <div key={index} className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-green-500 mr-2 flex-shrink-0" />
                  <span className="text-sm">{item}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="w-5 h-5 mr-2 text-blue-600" />
            Quick Insights
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 rounded-lg border-l-4 border-blue-500">
              <div className="font-medium text-blue-900 mb-1">Data Quality</div>
              <div className="text-sm text-blue-700">
                {edaResults.insights?.dataQuality?.[0] || 'Assessment completed'}
              </div>
            </div>
            <div className="p-4 bg-green-50 rounded-lg border-l-4 border-green-500">
              <div className="font-medium text-green-900 mb-1">ML Readiness</div>
              <div className="text-sm text-green-700">
                {mlValidationResult?.status === 'PROCEED' ? 
                  `✅ Ready (Score: ${mlValidationResult?.satisfaction_score}/100)` : 
                  'Check ML Validation tab for details'}
              </div>
            </div>
          </div>
        </div>

        {/* Simplified ML Status Card */}
        {(mlValidationResult || mlResults) && (
          <div className={`rounded-lg shadow p-6 border-l-4 ${
            mlValidationResult?.status === 'PROCEED' ? 'bg-green-50 border-green-500' :
            mlValidationResult?.status === 'PAUSE' ? 'bg-yellow-50 border-yellow-500' :
            'bg-blue-50 border-blue-500'
          }`}>
            <div className="flex items-center gap-3">
              <div className={`w-10 h-10 rounded-full flex items-center justify-center text-xl ${
                mlValidationResult?.status === 'PROCEED' ? 'bg-green-100 text-green-700' :
                mlValidationResult?.status === 'PAUSE' ? 'bg-yellow-100 text-yellow-700' :
                'bg-blue-100 text-blue-700'
              }`}>
                {mlValidationResult?.status === 'PROCEED' ? '✅' : 
                 mlValidationResult?.status === 'PAUSE' ? '⚠️' : '📊'}
              </div>
              <div>
                <h3 className={`text-xl font-bold ${
                  mlValidationResult?.status === 'PROCEED' ? 'text-green-800' :
                  mlValidationResult?.status === 'PAUSE' ? 'text-yellow-800' :
                  'text-blue-800'
                }`}>
                  {mlValidationResult?.status === 'PROCEED' ? 'Ready to Proceed!' :
                   mlValidationResult?.status === 'PAUSE' ? 'Review Required' :
                   'Analysis Complete'}
                </h3>
                <p className="text-sm text-gray-600">
                  Quality Score: {mlValidationResult?.satisfaction_score || mlResults?.satisfaction_score || 'N/A'}/100
                </p>
              </div>
            </div>
            {(mlValidationResult?.agent_answer || mlResults?.agent_answer) && (
              <div className="mt-4 p-3 bg-white rounded border">
                <div className="text-sm text-gray-700 line-clamp-3">
                  {(mlValidationResult?.agent_answer || mlResults?.agent_answer).split('\n')[0]}
                </div>
                <div className="text-xs text-blue-600 mt-1">View full report in ML Validation tab</div>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderEDA = () => {
    if (!edaResults) return null;

    return (
      <div className="space-y-6">
        {/* Missing Values Analysis */}
        {edaResults.missingValues && Object.keys(edaResults.missingValues).length > 0 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Missing Values Analysis</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {Object.entries(edaResults.missingValues)
                .filter(([col, info]) => info.count > 0)
                .map(([col, info]) => (
                  <div key={col} className={`p-4 rounded-lg border-l-4 ${
                    info.percentage > 50 ? 'bg-red-50 border-red-500' :
                    info.percentage > 20 ? 'bg-yellow-50 border-yellow-500' :
                    'bg-blue-50 border-blue-500'
                  }`}>
                    <div className="font-medium text-gray-800 truncate">{col}</div>
                    <div className="text-2xl font-bold mt-1">{info.percentage}%</div>
                    <div className="text-xs text-gray-600">{info.count} missing</div>
                  </div>
                ))
              }
              {Object.values(edaResults.missingValues).every(info => info.count === 0) && (
                <div className="col-span-full text-center p-4 bg-green-50 rounded-lg border border-green-200">
                  <span className="text-green-700 font-medium">✓ No missing values detected</span>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Numerical Features Summary</h3>
          <div className="overflow-x-auto">
            {edaResults.numericalSummary && Object.keys(edaResults.numericalSummary).length > 0 ? (
            <table className="min-w-full table-auto">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-4 py-2 text-left">Feature</th>
                  <th className="px-4 py-2 text-left">Count</th>
                  <th className="px-4 py-2 text-left">Mean</th>
                  <th className="px-4 py-2 text-left">Median</th>
                  <th className="px-4 py-2 text-left">Mode</th>
                  <th className="px-4 py-2 text-left">Std</th>
                  <th className="px-4 py-2 text-left">IQR</th>
                  <th className="px-4 py-2 text-left">Skewness</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(edaResults.numericalSummary).map(([col, stats]) => (
                  <tr key={col} className="border-t">
                    <td className="px-4 py-2 font-medium">{col}</td>
                    <td className="px-4 py-2">{stats.count}</td>
                    <td className="px-4 py-2">{stats.mean}</td>
                    <td className="px-4 py-2">{stats.median}</td>
                    <td className="px-4 py-2">{stats.mode || 'N/A'}</td>
                    <td className="px-4 py-2">{stats.std}</td>
                    <td className="px-4 py-2">{stats.iqr}</td>
                    <td className="px-4 py-2">{stats.skewness}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            ) : (
              <p className="text-gray-500 text-center py-4">No numerical features found in the dataset</p>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Categorical Features Summary</h3>
          {edaResults.objectSummary && Object.keys(edaResults.objectSummary).length > 0 ? (
          <div className="space-y-4">
            {Object.entries(edaResults.objectSummary).map(([col, stats]) => (
              <div key={col} className="border rounded p-4">
                <h4 className="font-medium mb-2">{col}</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>Unique: {stats.unique}</div>
                  <div>Entropy: {stats.entropy}</div>
                  <div>Top Value: {stats.topValues?.[0]?.value || 'N/A'}</div>
                  <div>Top %: {stats.topValues?.[0]?.percentage || 0}%</div>
                </div>
              </div>
            ))}
          </div>
          ) : (
            <p className="text-gray-500 text-center py-4">No categorical features found in the dataset</p>
          )}
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Correlation Matrix</h3>
          {Object.keys(edaResults.correlation).length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full table-auto">
                <thead>
                  <tr className="bg-gray-50">
                    <th className="px-4 py-2 text-left">Features</th>
                    {Object.keys(edaResults.correlation).map(col => (
                      <th key={col} className="px-4 py-2 text-left">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(edaResults.correlation).map(([col1, corrs]) => (
                    <tr key={col1} className="border-t">
                      <td className="px-4 py-2 font-medium">{col1}</td>
                      {Object.values(corrs).map((val, idx) => (
                        <td key={idx} className="px-4 py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            Math.abs(val) > 0.7 ? 'bg-red-100 text-red-800' :
                            Math.abs(val) > 0.5 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {val.toFixed(2)}
                          </span>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-gray-500">No correlations available (need at least 2 numeric columns)</p>
          )}
        </div>

        {/* Data Quality Insights */}
        {edaResults.insights && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4">Data Quality Insights</h3>
            <div className="space-y-4">
              {edaResults.insights.dataQuality && edaResults.insights.dataQuality.length > 0 && (
                <div>
                  <h4 className="font-medium text-blue-900 mb-2">📊 Data Quality</h4>
                  <div className="space-y-2">
                    {edaResults.insights.dataQuality.map((insight, idx) => (
                      <div key={idx} className="p-3 bg-blue-50 rounded-lg text-sm border-l-4 border-blue-400">
                        {insight}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {edaResults.insights.featureInsights && edaResults.insights.featureInsights.length > 0 && (
                <div>
                  <h4 className="font-medium text-green-900 mb-2">🔍 Feature Insights</h4>
                  <div className="space-y-2">
                    {edaResults.insights.featureInsights.slice(0, 5).map((insight, idx) => (
                      <div key={idx} className="p-3 bg-green-50 rounded-lg text-sm border-l-4 border-green-400">
                        {insight}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {edaResults.insights.correlationInsights && edaResults.insights.correlationInsights.length > 0 && (
                <div>
                  <h4 className="font-medium text-purple-900 mb-2">🔗 Correlation Insights</h4>
                  <div className="space-y-2">
                    {edaResults.insights.correlationInsights.slice(0, 3).map((insight, idx) => (
                      <div key={idx} className="p-3 bg-purple-50 rounded-lg text-sm border-l-4 border-purple-400">
                        {insight}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderMLValidation = () => {
    if (!mlResults) return null;

    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Brain className="w-5 h-5 mr-2 text-purple-600" />
            ML Validation Results
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-medium mb-2">Preprocessing Steps</h4>
              <div className="space-y-2">
                {mlResults.preprocessingSteps?.map((step, index) => (
                  <div key={index} className="p-3 bg-blue-50 rounded">
                    <div className="font-medium text-blue-800">{step.step}</div>
                    <div className="text-sm text-blue-600">{step.description}</div>
                    <div className="text-xs text-blue-500 mt-1">Priority: {step.priority}</div>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <h4 className="font-medium mb-2">Feature Engineering</h4>
              <div className="space-y-2">
                {mlResults.featureEngineering?.map((feat, index) => (
                  <div key={index} className="p-3 bg-green-50 rounded">
                    <div className="font-medium text-green-800">{feat.type}</div>
                    <div className="text-sm text-green-600">{feat.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Recommended Models</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {mlResults.modelRecommendations?.map((model, index) => (
              <div key={index} className="p-4 border rounded">
                <h4 className="font-medium text-lg mb-2">{model.algorithm}</h4>
                <div className="text-sm text-gray-600 mb-2">Type: {model.type}</div>
                <div className="space-y-1">
                  <div className="text-sm">
                    <span className="font-medium text-green-600">Pros:</span> {model.pros.join(', ')}
                  </div>
                  <div className="text-sm">
                    <span className="font-medium text-red-600">Cons:</span> {model.cons.join(', ')}
                  </div>
                  <div className="text-sm">
                    <span className="font-medium text-blue-600">Use Case:</span> {model.use_case}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Validation Metrics</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {mlResults.validationMetrics?.map((metric, index) => (
              <div key={index} className="text-center p-3 bg-gray-50 rounded">
                <div className="font-medium">{metric}</div>
              </div>
            ))}
          </div>
        </div>

        {mlResults.risksAndWarnings && mlResults.risksAndWarnings.length > 0 && (
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <AlertCircle className="w-5 h-5 mr-2 text-red-600" />
              Risks & Warnings
            </h3>
            <div className="space-y-3">
              {mlResults.risksAndWarnings.map((risk, index) => (
                <div key={index} className={`p-3 rounded border-l-4 ${
                  risk.level === 'High' ? 'border-red-500 bg-red-50' :
                  risk.level === 'Medium' ? 'border-yellow-500 bg-yellow-50' :
                  'border-blue-500 bg-blue-50'
                }`}>
                  <div className="font-medium">{risk.level} Risk</div>
                  <div className="text-sm">{risk.issue}</div>
                  <div className="text-sm text-gray-600 mt-1">{risk.description}</div>
                  <div className="text-sm font-medium mt-1">Mitigation: {risk.mitigation}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Performance Estimates</h3>
          <div className="p-4 bg-gray-50 rounded">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
              <div>
                <div className="text-2xl font-bold text-blue-600">{mlResults.performanceEstimates?.confidence || 'Unknown'}</div>
                <div className="text-sm text-gray-600">Confidence Level</div>
              </div>
              <div>
                <div className="text-lg font-medium text-green-600">{mlResults.performanceEstimates?.expected_accuracy || 'Unknown'}</div>
                <div className="text-sm text-gray-600">Expected Accuracy</div>
              </div>
              <div>
                <div className="text-lg font-medium text-purple-600">{mlResults.performanceEstimates?.data_sufficiency || 'Unknown'}</div>
                <div className="text-sm text-gray-600">Data Sufficiency</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderCode = () => {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center">
              <Code className="w-5 h-5 mr-2 text-blue-600" />
              Generated Python Code
            </h3>
            <div className="flex gap-2">
              <button
                onClick={() => setShowCode(!showCode)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                {showCode ? 'Hide Code' : 'Show Code'}
              </button>
              <button
                onClick={() => {
                  const code = (mlResults?.implementationCode?.full_pipeline || generateCode());
                  navigator.clipboard.writeText(code);
                  alert('Code copied to clipboard!');
                }}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Copy Code
              </button>
            </div>
          </div>
          
          {showCode && (
            <div className="space-y-4">
              <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <h4 className="text-green-400 mb-3 font-semibold"># Complete EDA & ML Pipeline</h4>
                <pre className="text-sm leading-relaxed">
                  {mlResults?.implementationCode?.full_pipeline || generateCode()}
                </pre>
              </div>

              {/* Code Sections */}
              {mlResults?.implementationCode && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {mlResults.implementationCode.eda_code && (
                    <div>
                      <h4 className="font-medium text-gray-800 mb-2">📊 EDA Code</h4>
                      <pre className="bg-gray-800 text-green-300 p-3 rounded text-xs overflow-x-auto max-h-40">
                        {mlResults.implementationCode.eda_code}
                      </pre>
                    </div>
                  )}
                  
                  {mlResults.implementationCode.preprocessing_code && (
                    <div>
                      <h4 className="font-medium text-gray-800 mb-2">🔧 Preprocessing Code</h4>
                      <pre className="bg-gray-800 text-blue-300 p-3 rounded text-xs overflow-x-auto max-h-40">
                        {mlResults.implementationCode.preprocessing_code}
                      </pre>
                    </div>
                  )}
                  
                  {mlResults.implementationCode.model_code && (
                    <div>
                      <h4 className="font-medium text-gray-800 mb-2">🤖 Model Code</h4>
                      <pre className="bg-gray-800 text-purple-300 p-3 rounded text-xs overflow-x-auto max-h-40">
                        {mlResults.implementationCode.model_code}
                      </pre>
                    </div>
                  )}
                  
                  {mlResults.implementationCode.validation_code && (
                    <div>
                      <h4 className="font-medium text-gray-800 mb-2">✅ Validation Code</h4>
                      <pre className="bg-gray-800 text-yellow-300 p-3 rounded text-xs overflow-x-auto max-h-40">
                        {mlResults.implementationCode.validation_code}
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Key Insights */}
          <div className="space-y-3 mt-6">
            <h4 className="font-semibold text-gray-800">📋 Key Implementation Notes:</h4>
            <div className="space-y-2">
              <div className="p-3 bg-blue-50 rounded border-l-4 border-blue-400 text-sm">
                <strong>Dataset:</strong> {edaResults?.shape?.rows || 'N/A'} rows × {edaResults?.shape?.columns || 'N/A'} columns
              </div>
              <div className="p-3 bg-green-50 rounded border-l-4 border-green-400 text-sm">
                <strong>Features:</strong> {edaResults?.numericColumns?.length || 0} numerical, {edaResults?.objectColumns?.length || 0} categorical
              </div>
              {mlValidationResult?.status === 'PROCEED' && (
                <div className="p-3 bg-green-50 rounded border-l-4 border-green-400 text-sm">
                  <strong>✅ Ready for ML:</strong> Dataset quality score {mlValidationResult.satisfaction_score}/100
                </div>
              )}
              {mlValidationResult?.goal_understanding?.target_column_guess && (
                <div className="p-3 bg-purple-50 rounded border-l-4 border-purple-400 text-sm">
                  <strong>Target Variable:</strong> {mlValidationResult.goal_understanding.target_column_guess}
                </div>
              )}
            </div>
          </div>

          {/* AI Insights */}
          {edaResults?.aiInsights && (
            <div className="mt-4">
              <h4 className="font-semibold text-gray-800 mb-2">🧠 AI-Generated Insights:</h4>
              <div className="p-4 bg-gradient-to-r from-purple-50 to-blue-50 rounded border border-purple-200">
                <div className="text-gray-700 whitespace-pre-wrap text-sm">
                  {edaResults.aiInsights}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderQuestions = () => {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <MessageSquare className="w-5 h-5 mr-2 text-blue-600" />
            Ask Questions About Your Data
          </h3>
          <div className="flex space-x-2 mb-4">
            <input
              type="text"
              value={currentQuestion}
              onChange={(e) => setCurrentQuestion(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && askQuestion()}
              placeholder="Ask anything about your dataset..."
              className="flex-1 p-2 border rounded"
            />
            <button
              onClick={askQuestion}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="space-y-4">
          {questions.length === 0 ? (
            <div className="bg-gray-50 rounded-lg p-6 text-center text-gray-500">
              <p>No questions asked yet. Start by asking a question about your dataset!</p>
            </div>
          ) : (
            questions.slice(-3).map((q, index) => (
              <div key={index} className="bg-white rounded-lg shadow p-6">
                <div className="mb-3">
                  <div className="font-medium text-gray-800">Q: {q.question}</div>
                  <div className="text-xs text-gray-500">{q.timestamp}</div>
                </div>
                <div className="text-gray-700">{q.answer}</div>
              </div>
            ))
          )}
          {questions.length > 3 && (
            <div className="text-center text-sm text-gray-500">
              Showing last 3 of {questions.length} questions
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto p-6">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">AI Validation Agent</h1>
          <p className="text-gray-600">Upload your dataset, specify your goal, and get comprehensive EDA + ML validation</p>
        </div>

        {currentStep === 'upload' && (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <Upload className="w-16 h-16 text-blue-600 mx-auto mb-4" />
            <h2 className="text-xl font-semibold mb-4">Upload Your Dataset</h2>
            <p className="text-gray-600 mb-6">Upload a CSV file to begin the validation process</p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Choose CSV File
            </button>
          </div>
        )}

        {currentStep === 'goal_input' && (
          <div className="bg-white rounded-lg shadow p-8">
            <div className="mb-6">
              <h2 className="text-xl font-semibold mb-2">Dataset Loaded Successfully!</h2>
              <p className="text-gray-600">Tell me your goal with this dataset</p>
            </div>
            <div className="space-y-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && startAnalysis()}
                placeholder="e.g., 'I want to predict customer churn' or 'I want to cluster products'"
                className="w-full p-3 border rounded"
              />
              <button
                onClick={startAnalysis}
                className="w-full py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center justify-center"
              >
                <Play className="w-5 h-5 mr-2" />
                Start ML Validation Process
              </button>
            </div>
          </div>
        )}

        {currentStep === 'processing' && (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <Loader2 className="w-16 h-16 text-blue-600 mx-auto mb-4 animate-spin" />
            <h2 className="text-xl font-semibold mb-4">Analyzing Your Dataset</h2>
            <p className="text-gray-600">Performing comprehensive EDA and ML validation...</p>
            <div className="mt-6 space-y-2">
              <div className="flex items-center">
                <CheckCircle className="w-5 h-5 text-green-500 mr-2" />
                <span>Goal Detection: {detectedGoal?.description}</span>
              </div>
              <div className="flex items-center">
                <Loader2 className="w-5 h-5 text-blue-500 mr-2 animate-spin" />
                <span>Running EDA Analysis...</span>
              </div>
              <div className="flex items-center text-gray-400">
                <div className="w-5 h-5 mr-2"></div>
                <span>Preparing ML Validation...</span>
              </div>
            </div>
          </div>
        )}

        {currentStep === 'results' && (
          <div className="space-y-6">
            {/* Analysis Complete Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg shadow-lg p-6">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center text-3xl">
                  🎯
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Analysis Complete</h2>
                  <p className="text-blue-100">Comprehensive validation results ready</p>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex space-x-1 mb-4">
                {[
                  { id: 'overview', label: 'Overview', icon: Database },
                  { id: 'eda', label: 'Exploratory Analysis', icon: BarChart3 },
                  { id: 'ml', label: 'ML Validation', icon: Brain },
                  { id: 'code', label: 'Code & Insights', icon: Code },
                  { id: 'questions', label: 'Ask Questions', icon: MessageSquare }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center px-4 py-2 rounded transition-colors ${
                      activeTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    }`}
                  >
                    <tab.icon className="w-4 h-4 mr-2" />
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="min-h-96">
                {activeTab === 'overview' && renderOverview()}
                {activeTab === 'eda' && renderEDA()}
                {activeTab === 'ml' && renderMLValidation()}
                {activeTab === 'code' && renderCode()}
                {activeTab === 'questions' && renderQuestions()}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ValidationAgenticAI;

  const generateInsights = (data) => {
    const insights = {
      dataQuality: [],
      featureInsights: [],
      correlationInsights: [],
      distributionInsights: [],
      actionableRecommendations: []
    };

    // Data Quality Insights
    const totalMissing = Object.values(data.missingValues).reduce((sum, v) => sum + v.count, 0);
    const missingPercentage = ((totalMissing / (data.shape.rows * data.shape.columns)) * 100).toFixed(2);
    
    if (totalMissing === 0) {
      insights.dataQuality.push('✅ Perfect data completeness - no missing values detected');
    } else if (parseFloat(missingPercentage) < 5) {
      insights.dataQuality.push(`✅ Excellent data quality - only ${missingPercentage}% missing values`);
    } else if (parseFloat(missingPercentage) < 20) {
      insights.dataQuality.push(`⚠️ Moderate missing data (${missingPercentage}%) - imputation recommended`);
    } else {
      insights.dataQuality.push(`❌ High missing data (${missingPercentage}%) - careful preprocessing required`);
    }

    if (data.shape.rows < 50) {
      insights.dataQuality.push(`⚠️ Limited sample size (${data.shape.rows} rows) - results may not be statistically robust`);
    } else if (data.shape.rows < 200) {
      insights.dataQuality.push(`✓ Adequate sample size (${data.shape.rows} rows) for basic analysis`);
    } else {
      insights.dataQuality.push(`✅ Good sample size (${data.shape.rows} rows) for reliable analysis`);
    }

    // Feature Insights - Variance and IQR Analysis
    const lowVarianceFeatures = [];
    const highVarianceFeatures = [];
    const wideIQRFeatures = [];
    
    Object.entries(data.numericalSummary).forEach(([col, stats]) => {
      const variance = parseFloat(stats.variance);
      const cv = parseFloat(stats.coefficientOfVariation);
      const iqr = parseFloat(stats.iqr);
      const range = parseFloat(stats.range);
      const iqrToRangeRatio = (iqr / range) * 100;
      
      // Low variance features
      if (cv < 10) {
        lowVarianceFeatures.push(`${col} (CV: ${cv}%)`);
      }
      
      // High variance features
      if (cv > 100) {
        highVarianceFeatures.push(`${col} (CV: ${cv}%)`);
      }
      
      // Wide IQR features (potential outliers or high spread)
      if (iqrToRangeRatio < 30) {
        wideIQRFeatures.push(`${col} (IQR: ${stats.iqr}, Range: ${stats.range})`);
      }
    });

    if (lowVarianceFeatures.length > 0) {
      insights.featureInsights.push(`📉 Low variance features (may not add much information): ${lowVarianceFeatures.join(', ')}`);
    }
    
    if (highVarianceFeatures.length > 0) {
      insights.featureInsights.push(`📈 High variance features (scaling strongly recommended): ${highVarianceFeatures.join(', ')}`);
    }
    
    if (wideIQRFeatures.length > 0) {
      insights.featureInsights.push(`📊 Features with wide spread (check for outliers): ${wideIQRFeatures.join(', ')}`);
    }

    // Unique values insights
    const lowCardinalityNumerical = [];
    const highCardinalityNumerical = [];
    
    Object.entries(data.numericalSummary).forEach(([col, stats]) => {
      const uniqueRatio = (stats.unique / stats.count) * 100;
      
      if (stats.unique <= 10) {
        lowCardinalityNumerical.push(`${col} (${stats.unique} unique values)`);
      } else if (uniqueRatio > 95) {
        highCardinalityNumerical.push(`${col} (${stats.unique} unique values)`);
      }
    });

    if (lowCardinalityNumerical.length > 0) {
      insights.featureInsights.push(`🔢 Numerical features with low cardinality (consider treating as categorical): ${lowCardinalityNumerical.join(', ')}`);
    }

    if (highCardinalityNumerical.length > 0) {
      insights.featureInsights.push(`🆔 High cardinality numerical features (may be identifiers): ${highCardinalityNumerical.join(', ')}`);
    }

    // Correlation Insights
    if (data.numericColumns.length > 1) {
      const strongPositive = [];
      const strongNegative = [];
      const moderate = [];
      
      Object.entries(data.correlation).forEach(([col1, corrs]) => {
        Object.entries(corrs).forEach(([col2, val]) => {
          if (col1 < col2) { // Avoid duplicates
            if (val > 0.8) {
              strongPositive.push(`${col1} ↔ ${col2} (${val})`);
            } else if (val < -0.8) {
              strongNegative.push(`${col1} ↔ ${col2} (${val})`);
            } else if (Math.abs(val) > 0.5 && Math.abs(val) <= 0.8) {
              moderate.push(`${col1} ↔ ${col2} (${val})`);
            }
          }
        });
      });

      if (strongPositive.length > 0) {
        insights.correlationInsights.push(`✅ Strong positive correlations found: ${strongPositive.slice(0, 3).join(', ')}${strongPositive.length > 3 ? ` and ${strongPositive.length - 3} more` : ''}`);
      }
      
      if (strongNegative.length > 0) {
        insights.correlationInsights.push(`⚠️ Strong negative correlations found: ${strongNegative.slice(0, 3).join(', ')}${strongNegative.length > 3 ? ` and ${strongNegative.length - 3} more` : ''}`);
      }
      
      if (moderate.length > 0) {
        insights.correlationInsights.push(`📊 Moderate correlations detected in ${moderate.length} feature pairs`);
      }

      if (strongPositive.length === 0 && strongNegative.length === 0 && moderate.length === 0) {
        insights.correlationInsights.push(`ℹ️ Weak correlations overall - features are mostly independent`);
      }
    }

    // Distribution Insights
    const normalDist = [];
    const skewedDist = [];
    const highlySkewedDist = [];
    
    Object.entries(data.numericalSummary).forEach(([col, stats]) => {
      const skew = parseFloat(stats.skewness);
      
      if (Math.abs(skew) < 0.5) {
        normalDist.push(col);
      } else if (Math.abs(skew) < 1) {
        skewedDist.push(`${col} (skew: ${stats.skewness})`);
      } else {
        highlySkewedDist.push(`${col} (skew: ${stats.skewness})`);
      }
    });

    if (normalDist.length > 0) {
      insights.distributionInsights.push(`📊 Approximately normal distributions: ${normalDist.join(', ')}`);
    }
    
    if (skewedDist.length > 0) {
      insights.distributionInsights.push(`📈 Moderately skewed features (may need transformation): ${skewedDist.join(', ')}`);
    }
    
    if (highlySkewedDist.length > 0) {
      insights.distributionInsights.push(`⚠️ Highly skewed features (log/power transformation recommended): ${highlySkewedDist.join(', ')}`);
    }

    // Outlier insights
    const featuresWithOutliers = Object.entries(data.numericalSummary)
      .filter(([_, stats]) => stats.outliers > 0)
      .map(([col, stats]) => `${col} (${stats.outliers} outliers, ${stats.outliersPercentage}%)`);
    
    if (featuresWithOutliers.length > 0) {
      insights.distributionInsights.push(`🔍 Features with outliers detected: ${featuresWithOutliers.join(', ')}`);
    }

    // Actionable Recommendations based on goal
    if (data.goal.type === 'supervised') {
      insights.actionableRecommendations.push('🎯 For Supervised Learning:');
      
      const strongPositive = [];
      Object.entries(data.correlation).forEach(([col1, corrs]) => {
        Object.entries(corrs).forEach(([col2, val]) => {
          if (col1 < col2 && val > 0.8) {
            strongPositive.push(`${col1} ↔ ${col2}`);
          }
        });
      });
      
      if (strongPositive.length > 2) {
        insights.actionableRecommendations.push('  • Remove redundant features with high correlation (>0.8) to avoid multicollinearity');
      }
      
      if (highlySkewedDist.length > 0) {
        insights.actionableRecommendations.push('  • Apply log/sqrt transformation to highly skewed features before modeling');
      }
      
      if (highVarianceFeatures.length > 0) {
        insights.actionableRecommendations.push('  • Normalize/standardize features to prevent dominance by high-variance features');
      }
      
      const outlierPercent = featuresWithOutliers.length / data.numericColumns.length * 100;
      if (outlierPercent > 30) {
        insights.actionableRecommendations.push('  • Consider robust scaling or outlier treatment (>30% features affected)');
      }
      
    } else if (data.goal.type === 'unsupervised') {
      insights.actionableRecommendations.push('🔍 For Unsupervised Learning:');
      
      insights.actionableRecommendations.push('  • Apply StandardScaler or MinMaxScaler - clustering is sensitive to scale');
      
      if (data.numericColumns.length > 10) {
        insights.actionableRecommendations.push(`  • Use PCA for dimensionality reduction (${data.numericColumns.length} features detected)`);
      }
      
      if (featuresWithOutliers.length > 0) {
        insights.actionableRecommendations.push('  • Handle outliers carefully - they may represent important clusters or anomalies');
      }
      
      if (lowVarianceFeatures.length > 0) {
        insights.actionableRecommendations.push('  • Remove low-variance features as they\'t contribute to clustering');
      }
      
    } else {
      insights.actionableRecommendations.push('📊 General Recommendations:');
      
      if (totalMissing > 0) {
        insights.actionableRecommendations.push('  • Handle missing values through imputation or removal');
      }
      
      if (highVarianceFeatures.length > 0) {
        insights.actionableRecommendations.push('  • Consider feature scaling for machine learning tasks');
      }
      
      if (data.shape.columns > 20) {
        insights.actionableRecommendations.push('  • Consider feature selection to reduce dimensionality');
      }
    }

    return insights;
  };

  const performAdvancedEDA = async (csvText, goal) => {
    setIsProcessing(true);
    
    try {
      // Call the backend API
      const response = await fetch('/validation/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          csv_text: csvText,
          goal: goal
        })
      });
      
      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }
      
      const result = await response.json();
      
      if (result.status === 'success') {
        setEdaResults(result.result);
        setIsProcessing(false);
        return result.result;
      } else {
        throw new Error('Analysis failed');
      }
    } catch (error) {
      console.error('EDA API call failed:', error);
      setMessages(prev => [...prev, { 
        type: 'agent', 
        content: `❌ Error performing analysis: ${error.message}. Please try again.` 
      }]);
      setIsProcessing(false);
      return null;
    }
  };

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (!uploadedFile) return;

    if (!uploadedFile.name.endsWith('.csv')) {
      setMessages(prev => [...prev, 
        { type: 'user', content: `Uploaded: ${uploadedFile.name}` },
        { type: 'agent', content: '❌ Please upload a CSV file only.' }
      ]);
      return;
    }

    setFile(uploadedFile);
    setMessages(prev => [...prev, { type: 'user', content: `Uploaded: ${uploadedFile.name}` }]);

    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const csvText = event.target.result;
        setDataset(csvText); // Store raw CSV text
        
        // Quick parse for display
        const lines = csvText.split('\n').filter(line => line.trim());
        const headers = lines[0].split(',').map(h => h.trim());
        const rows = lines.slice(1);
        
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: `✅ Dataset loaded successfully!\n\n📊 Dataset Info:\n- Rows: ${rows.length}\n- Columns: ${headers.length}\n- Column Names: ${headers.join(', ')}\n\n🎯 What's your goal with this dataset?\n\nExamples:\n• "I want to predict sales"\n• "I want to cluster customers"\n• "Just explore and analyze the data"\n\nTell me your goal, and I'll tailor the analysis accordingly!` 
        }]);
        setAwaitingGoal(true);
      } catch (error) {
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: '❌ Error parsing CSV file. Please ensure it\'s properly formatted.' 
        }]);
      }
    };
    reader.readAsText(uploadedFile);
  };

  const handleSendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
    setInput('');

    // Handle goal detection
    if (awaitingGoal && dataset) {
      setAwaitingGoal(false);
      const detectedGoal = detectGoal(userMessage);
      setUserGoal(detectedGoal);
      
      setMessages(prev => [...prev, { 
        type: 'agent', 
        content: `Perfect! I've detected your goal:\n\n🎯 **${detectedGoal.description}**\n\nI'll focus on:\n${detectedGoal.focus.map((f, i) => `${i + 1}. ${f}`).join('\n')}\n\n🚀 Starting advanced validation and EDA...` 
      }]);
      
      const result = await performAdvancedEDA(dataset, detectedGoal);
      
      if (result) {
        let conclusionMsg = `✅ **Analysis Complete!**\n\n`;
        conclusionMsg += `${result.isValid ? '✅ Dataset Validation: PASSED' : '⚠️ Dataset Validation: ISSUES DETECTED'}\n\n`;
        
        if (result.recommendations && result.recommendations.length > 0) {
          conclusionMsg += `**Key Recommendations:**\n${result.recommendations.join('\n')}\n\n`;
        }
        
        // Add comprehensive insights
        if (result.insights) {
          conclusionMsg += `📊 **Comprehensive Insights Generated!**\n\n`;
          conclusionMsg += `The analysis has uncovered ${result.insights.dataQuality.length + result.insights.featureInsights.length + result.insights.correlationInsights.length + result.insights.distributionInsights.length} key findings across:\n`;
          conclusionMsg += `• Data Quality (${result.insights.dataQuality.length} insights)\n`;
          conclusionMsg += `• Feature Analysis (${result.insights.featureInsights.length} insights)\n`;
          conclusionMsg += `• Correlations (${result.insights.correlationInsights.length} insights)\n`;
          conclusionMsg += `• Distributions (${result.insights.distributionInsights.length} insights)\n`;
          conclusionMsg += `• Actionable Recommendations (${result.insights.actionableRecommendations.length} items)\n\n`;
        }
        
        conclusionMsg += `📊 View detailed insights in the right panel!\n\n`;
        conclusionMsg += `💡 You can:\n• Ask "show insights" for detailed findings\n• Type "show code" for Python implementation\n• Ask specific questions about your data`;
        
        setMessages(prev => [...prev, { type: 'agent', content: conclusionMsg }]);
      }
      return;
    }

    // Handle code request
    if (userMessage.toLowerCase().includes('show code') || userMessage.toLowerCase().includes('python code')) {
      if (edaResults) {
        setShowCode(true);
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: '📝 I\'ve opened the Python pandas code in the right panel. This shows you exactly how to perform this analysis using pandas, numpy, and other libraries!' 
        }]);
      } else {
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: 'Please complete the analysis first, then I can show you the Python code!' 
        }]);
      }
      return;
    }

    // Handle data queries using API
    if (edaResults) {
      try {
        const response = await fetch('/validation/question', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            question: userMessage,
            eda_results: edaResults
          })
        });
        
        if (response.ok) {
          const result = await response.json();
          setMessages(prev => [...prev, { 
            type: 'agent', 
            content: result.answer 
          }]);
        } else {
          setMessages(prev => [...prev, { 
            type: 'agent', 
            content: '❌ Sorry, I couldn\'t process your question. Please try rephrasing.' 
          }]);
        }
      } catch (error) {
        console.error('Question API call failed:', error);
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: '❌ Error connecting to analysis service. Please try again.' 
        }]);
      }
      return;
    }
        if (edaResults.numericColumns.length > 1) {
          const strongCorr = [];
          Object.entries(edaResults.correlation).forEach(([col1, correlations]) => {
            Object.entries(correlations).forEach(([col2, value]) => {
              if (col1 !== col2 && Math.abs(value) > 0.7) {
                strongCorr.push(`${col1} ↔ ${col2}: ${value}`);
              }
            });
          });
          response = strongCorr.length > 0 
            ? `🔗 **Strong Correlations (>0.7):**\n${strongCorr.slice(0, 5).join('\n')}`
            : '📊 No strong correlations (>0.7) found.';
        } else {
          response = 'ℹ️ Need at least 2 numerical columns for correlation analysis.';
        }
      }
      else if (userMessage.toLowerCase().includes('recommend') || userMessage.toLowerCase().includes('suggestion')) {
        response = `💡 **Recommendations for ${edaResults.goal.description}:**\n\n`;
        if (edaResults.recommendations.length > 0) {
          response += edaResults.recommendations.join('\n\n');
        } else {
          response += '✅ Your dataset looks good! No major issues detected.';
        }
      }
      else if (userMessage.toLowerCase().includes('summary') || userMessage.toLowerCase().includes('overview')) {
        response = `📈 **Dataset Overview:**\n\n`;
        response += `**Goal:** ${edaResults.goal.description}\n`;
        response += `**Validation:** ${edaResults.isValid ? '✅ Passed' : '⚠️ Issues detected'}\n`;
        response += `**Data Quality:** ${edaResults.validationChecks.dataQuality}\n`;
        response += `**Sample Size:** ${edaResults.validationChecks.sufficientSamples}\n`;
        response += `**Missing Data:** ${edaResults.validationChecks.missingDataLevel}\n\n`;
        response += `**Features:**\n• Numerical: ${edaResults.numericColumns.length}\n• Categorical: ${edaResults.objectColumns.length}`;
      }
      else {
        response = `I can help you with:\n\n`;
        response += `📊 **Analysis:** "show insights", "variance and IQR", "unique values"\n`;
        response += `📏 **Data Info:** "show columns", "dataset shape"\n`;
        response += `❓ **Quality:** "missing values", "outliers"\n`;
        response += `🔗 **Patterns:** "correlations", "summary"\n`;
        response += `💡 **Actions:** "recommendations"\n`;
        response += `💻 **Code:** "show code" or "python code"\n\n`;
        response += `Just ask naturally!`;
      }
      
      setMessages(prev => [...prev, { type: 'agent', content: response }]);
    } else {
      setMessages(prev => [...prev, { 
        type: 'agent', 
        content: 'Please upload a dataset first to start the analysis!' 
      }]);
    }
  };

  const getPythonCode = () => {
    if (!edaResults) return '';
    
    const goalType = edaResults.goal.type;
    
    return `# Advanced EDA and Validation with Python Pandas
# Goal: ${edaResults.goal.description}
# This code performs comprehensive exploratory data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ============================================
# STEP 1: LOAD THE DATASET
# ============================================

# Load your CSV file
df = pd.read_csv('your_dataset.csv')

print("Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================
# STEP 2: BASIC INFORMATION
# ============================================

print("\\n" + "="*50)
print("BASIC INFORMATION")
print("="*50)

# Display first few rows
print("\\nFirst 5 rows:")
print(df.head())

# Dataset info
print("\\nDataset Info:")
df.info()

# Column types
print("\\nColumn Data Types:")
print(df.dtypes)

# ============================================
# STEP 3: MISSING VALUES ANALYSIS
# ============================================

print("\\n" + "="*50)
print("MISSING VALUES ANALYSIS")
print("="*50)

# Calculate missing values
missing_values = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Percentage': (df.isnull().sum() / len(df)) * 100
})

# Filter columns with missing values
missing_values = missing_values[missing_values['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_values) > 0:
    print("\\nColumns with missing values:")
    print(missing_values.to_string(index=False))
    
    # Handle missing values
    print("\\nHandling missing values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Fill numerical columns with median
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  ✓ {col}: Filled with median ({df[col].median():.2f})")
            else:
                # Fill categorical columns with mode
                df[col].fillna(df[col].mode()[0], inplace=True)
                print(f"  ✓ {col}: Filled with mode ({df[col].mode()[0]})")
else:
    print("\\n✅ No missing values found!")

# ============================================
# STEP 4: STATISTICAL SUMMARY
# ============================================

print("\\n" + "="*50)
print("STATISTICAL SUMMARY")
print("="*50)

# Numerical columns summary
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numerical_cols:
    print(f"\\nNumerical Columns ({len(numerical_cols)}):")
    print(df[numerical_cols].describe())
    
    # Additional statistics including variance, IQR, and unique values
    print("\\nAdvanced Statistics (Variance, IQR, Unique Values):")
  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat Panel */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center">
            <Brain className="mr-2 text-blue-600" />
            Validation Agentic AI
          </h1>
          <p className="text-gray-600 mt-1">Advanced dataset validation and EDA with AI-powered insights</p>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-3xl rounded-lg p-3 ${
                message.type === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-white border border-gray-200 text-gray-800'
              }`}>
                <pre className="whitespace-pre-wrap font-sans">{message.content}</pre>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex space-x-2">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              ref={fileInputRef}
              className="hidden"
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <Upload className="mr-2" size={16} />
              Upload CSV
            </button>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Ask questions about your data..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isProcessing}
            />
            <button
              onClick={handleSendMessage}
              disabled={isProcessing || !input.trim()}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              {isProcessing ? <Loader2 className="animate-spin" size={16} /> : <Send size={16} />}
            </button>
          </div>
        </div>
      </div>

      {/* Results Panel */}
      <div className="w-96 bg-white border-l border-gray-200 overflow-y-auto">
        {edaResults ? (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <BarChart3 className="mr-2 text-green-600" />
              Analysis Results
            </h2>

            {/* Dataset Overview */}
            <div className="mb-6">
              <h3 className="font-medium mb-2">📊 Dataset Overview</h3>
              <div className="bg-gray-50 p-3 rounded">
                <p><strong>Shape:</strong> {edaResults.shape.rows} rows × {edaResults.shape.columns} columns</p>
                <p><strong>Size:</strong> {edaResults.size} cells</p>
                <p><strong>Validation:</strong> {edaResults.isValid ? 
                  <span className="text-green-600">✅ Passed</span> : 
                  <span className="text-red-600">⚠️ Issues</span>
                }</p>
              </div>
            </div>

            {/* Numerical Summary */}
            {Object.keys(edaResults.numericalSummary).length > 0 && (
              <div className="mb-6">
                <h3 className="font-medium mb-2">🔢 Numerical Features</h3>
                <div className="space-y-2">
                  {Object.entries(edaResults.numericalSummary).slice(0, 3).map(([col, stats]) => (
                    <div key={col} className="bg-gray-50 p-3 rounded">
                      <p className="font-medium">{col}</p>
                      <div className="text-sm text-gray-600 grid grid-cols-2 gap-1">
                        <span>Mean: {stats.mean}</span>
                        <span>Median: {stats.median}</span>
                        <span>Mode: {stats.mode || 'N/A'}</span>
                        <span>Std: {stats.std}</span>
                        <span>IQR: {stats.iqr}</span>
                        <span>Skew: {stats.skewness}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Categorical Summary */}
            {Object.keys(edaResults.objectSummary).length > 0 && (
              <div className="mb-6">
                <h3 className="font-medium mb-2">📋 Categorical Features</h3>
                <div className="space-y-2">
                  {Object.entries(edaResults.objectSummary).slice(0, 3).map(([col, stats]) => (
                    <div key={col} className="bg-gray-50 p-3 rounded">
                      <p className="font-medium">{col}</p>
                      <p className="text-sm text-gray-600">
                        {stats.unique} unique values
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Insights */}
            {edaResults.insights && (
              <div className="mb-6">
                <h3 className="font-medium mb-2">💡 Key Insights</h3>
                <div className="space-y-2">
                  {edaResults.insights.dataQuality.slice(0, 2).map((insight, i) => (
                    <p key={i} className="text-sm bg-blue-50 p-2 rounded">{insight}</p>
                  ))}
                  {edaResults.insights.featureInsights.slice(0, 2).map((insight, i) => (
                    <p key={i} className="text-sm bg-green-50 p-2 rounded">{insight}</p>
                  ))}
                </div>
              </div>
            )}

            {/* Recommendations */}
            {edaResults.recommendations && edaResults.recommendations.length > 0 && (
              <div className="mb-6">
                <h3 className="font-medium mb-2">🎯 Recommendations</h3>
                <div className="space-y-1">
                  {edaResults.recommendations.slice(0, 3).map((rec, i) => (
                    <p key={i} className="text-sm bg-yellow-50 p-2 rounded">{rec}</p>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : showCode ? (
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4 flex items-center">
              <Code className="mr-2 text-purple-600" />
              Python Implementation
            </h2>
            <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded overflow-x-auto">
              {getPythonCode()}
            </pre>
          </div>
        ) : (
          <div className="p-4 text-center text-gray-500">
            <FileText className="mx-auto mb-2" size={48} />
            <p>Upload a CSV file and start analyzing!</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ValidationAgenticAI;
