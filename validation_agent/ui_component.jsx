import React, { useState, useRef, useEffect } from 'react';
import { Upload, Send, Code, BarChart3, CheckCircle, XCircle, Loader2, FileText, TrendingUp, AlertCircle, Target, Brain } from 'lucide-react';

const ValidationAgenticAI = () => {
  const [messages, setMessages] = useState([
    { type: 'agent', content: 'Hello! I\'m your Validation Agentic AI. Upload your dataset (CSV format) and tell me your goal - I\'ll help you validate and analyze it with tailored insights!' }
  ]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [edaResults, setEdaResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [awaitingGoal, setAwaitingGoal] = useState(false);
  const [userGoal, setUserGoal] = useState(null);
  const [showCode, setShowCode] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, edaResults]);

  const parseCSV = (text) => {
    const lines = text.split('\n').filter(line => line.trim());
    const headers = lines[0].split(',').map(h => h.trim());
    const rows = lines.slice(1).map(line => {
      const values = line.split(',').map(v => v.trim());
      const row = {};
      headers.forEach((header, i) => {
        row[header] = values[i];
      });
      return row;
    });
    return { headers, rows };
  };

  const detectGoal = (userInput) => {
    const input = userInput.toLowerCase();
    
    // Supervised learning keywords
    const supervisedKeywords = ['predict', 'classification', 'regression', 'forecast', 'supervised', 
                                'target', 'label', 'outcome', 'predict price', 'predict sales',
                                'classify', 'categorize'];
    
    // Unsupervised learning keywords
    const unsupervisedKeywords = ['cluster', 'segment', 'pattern', 'group', 'unsupervised', 
                                  'anomaly', 'outlier', 'similarity', 'discover'];
    
    // EDA/Analysis keywords
    const edaKeywords = ['analyze', 'explore', 'understand', 'insights', 'statistics', 
                         'visualize', 'eda', 'exploratory'];
    
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

  const performAdvancedEDA = async (data, goal) => {
    setIsProcessing(true);
    await new Promise(resolve => setTimeout(resolve, 2000));

    const { headers, rows } = data;
    const shape = { rows: rows.length, columns: headers.length };
    
    // Column type detection
    const columnTypes = {};
    const numericColumns = [];
    const objectColumns = [];
    
    headers.forEach(col => {
      const values = rows.map(row => row[col]).filter(v => v);
      const isNumeric = values.every(v => !isNaN(parseFloat(v)));
      columnTypes[col] = isNumeric ? 'numerical' : 'categorical';
      if (isNumeric) numericColumns.push(col);
      else objectColumns.push(col);
    });

    // Missing values analysis
    const missingValues = {};
    headers.forEach(col => {
      const missing = rows.filter(row => !row[col] || row[col] === '').length;
      missingValues[col] = { 
        count: missing, 
        percentage: ((missing / rows.length) * 100).toFixed(2),
        severity: missing / rows.length > 0.5 ? 'High' : missing / rows.length > 0.2 ? 'Medium' : 'Low'
      };
    });

    // Unique values
    const uniqueValues = {};
    headers.forEach(col => {
      const unique = new Set(rows.map(row => row[col])).size;
      uniqueValues[col] = unique;
    });

    // Advanced numerical summary with variance, IQR, and unique values
    const numericalSummary = {};
    numericColumns.forEach(col => {
      const values = rows.map(row => parseFloat(row[col])).filter(v => !isNaN(v));
      if (values.length > 0) {
        const sorted = values.sort((a, b) => a - b);
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
        const std = Math.sqrt(variance);
        const skewness = values.reduce((sum, val) => sum + Math.pow((val - mean) / std, 3), 0) / values.length;
        
        // Quartiles and IQR calculation
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        const lowerBound = q1 - 1.5 * iqr;
        const upperBound = q3 + 1.5 * iqr;
        const outliers = values.filter(v => v < lowerBound || v > upperBound).length;
        
        // Unique values count
        const uniqueCount = new Set(values).size;
        
        // Range
        const range = Math.max(...values) - Math.min(...values);
        
        numericalSummary[col] = {
          count: values.length,
          unique: uniqueCount,
          mean: mean.toFixed(2),
          median: sorted[Math.floor(sorted.length / 2)].toFixed(2),
          std: std.toFixed(2),
          variance: variance.toFixed(2),
          min: Math.min(...values).toFixed(2),
          max: Math.max(...values).toFixed(2),
          range: range.toFixed(2),
          q1: q1.toFixed(2),
          q3: q3.toFixed(2),
          iqr: iqr.toFixed(2),
          lowerBound: lowerBound.toFixed(2),
          upperBound: upperBound.toFixed(2),
          skewness: skewness.toFixed(2),
          outliers: outliers,
          outliersPercentage: ((outliers / values.length) * 100).toFixed(2),
          coefficientOfVariation: ((std / mean) * 100).toFixed(2),
          isNormalDist: Math.abs(skewness) < 0.5 ? 'Approximately Normal' : Math.abs(skewness) < 1 ? 'Moderately Skewed' : 'Highly Skewed'
        };
      }
    });

    // Categorical summary
    const objectSummary = {};
    objectColumns.forEach(col => {
      const values = rows.map(row => row[col]).filter(v => v);
      const frequency = {};
      values.forEach(v => {
        frequency[v] = (frequency[v] || 0) + 1;
      });
      const topValues = Object.entries(frequency)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
      
      const uniqueCount = Object.keys(frequency).length;
      const entropy = -Object.values(frequency).reduce((sum, count) => {
        const p = count / values.length;
        return sum + (p * Math.log2(p));
      }, 0);
      
      objectSummary[col] = {
        count: values.length,
        unique: uniqueCount,
        uniquePercentage: ((uniqueCount / values.length) * 100).toFixed(2),
        entropy: entropy.toFixed(2),
        topValues: topValues.map(([val, count]) => ({ 
          value: val, 
          count, 
          percentage: ((count / values.length) * 100).toFixed(2)
        })),
        isPotentialTarget: uniqueCount >= 2 && uniqueCount <= 20 && goal.type === 'supervised'
      };
    });

    // Correlation analysis
    const correlation = {};
    if (numericColumns.length > 1) {
      numericColumns.forEach(col1 => {
        correlation[col1] = {};
        numericColumns.forEach(col2 => {
          const values1 = rows.map(row => parseFloat(row[col1])).filter(v => !isNaN(v));
          const values2 = rows.map(row => parseFloat(row[col2])).filter(v => !isNaN(v));
          
          if (values1.length > 0 && values2.length > 0) {
            const mean1 = values1.reduce((a, b) => a + b, 0) / values1.length;
            const mean2 = values2.reduce((a, b) => a + b, 0) / values2.length;
            
            let numerator = 0;
            let sum1 = 0;
            let sum2 = 0;
            
            for (let i = 0; i < Math.min(values1.length, values2.length); i++) {
              numerator += (values1[i] - mean1) * (values2[i] - mean2);
              sum1 += Math.pow(values1[i] - mean1, 2);
              sum2 += Math.pow(values2[i] - mean2, 2);
            }
            
            const corr = numerator / Math.sqrt(sum1 * sum2);
            correlation[col1][col2] = isNaN(corr) ? 0 : parseFloat(corr.toFixed(2));
          }
        });
      });
    }

    // Goal-specific validation
    const validationChecks = {
      hasData: rows.length > 0,
      hasColumns: headers.length > 0,
      noEmptyColumns: headers.every(col => rows.some(row => row[col])),
      missingDataLevel: Object.values(missingValues).every(m => parseFloat(m.percentage) < 50) ? 'Acceptable' : 'High',
      dataQuality: rows.length > 30 ? 'Good' : rows.length > 10 ? 'Fair' : 'Limited',
      sufficientSamples: rows.length >= 100 ? 'Excellent' : rows.length >= 30 ? 'Good' : 'Limited'
    };

    // Goal-specific recommendations
    const recommendations = [];
    
    if (goal.type === 'supervised') {
      const potentialTargets = objectColumns.filter(col => 
        objectSummary[col].unique >= 2 && objectSummary[col].unique <= 20
      );
      
      if (potentialTargets.length > 0) {
        recommendations.push(`🎯 Potential target variables: ${potentialTargets.join(', ')}`);
      }
      
      const highCorrelations = [];
      Object.entries(correlation).forEach(([col1, corrs]) => {
        Object.entries(corrs).forEach(([col2, val]) => {
          if (col1 !== col2 && Math.abs(val) > 0.8) {
            highCorrelations.push(`${col1} ↔ ${col2} (${val})`);
          }
        });
      });
      
      if (highCorrelations.length > 0) {
        recommendations.push(`⚠️ High multicollinearity detected - consider feature selection`);
      }
      
      const imbalancedCategorical = objectColumns.filter(col => {
        const topValue = objectSummary[col].topValues[0];
        return topValue && parseFloat(topValue.percentage) > 80;
      });
      
      if (imbalancedCategorical.length > 0) {
        recommendations.push(`⚖️ Imbalanced classes detected in: ${imbalancedCategorical.join(', ')}`);
      }
    }
    
    if (goal.type === 'unsupervised') {
      const highVarianceFeatures = numericColumns.filter(col => 
        parseFloat(numericalSummary[col].coefficientOfVariation) > 100
      );
      
      if (highVarianceFeatures.length > 0) {
        recommendations.push(`📊 High variance features (consider scaling): ${highVarianceFeatures.join(', ')}`);
      }
      
      const outlierColumns = numericColumns.filter(col => 
        parseFloat(numericalSummary[col].outliersPercentage) > 5
      );
      
      if (outlierColumns.length > 0) {
        recommendations.push(`🔍 Outliers detected in: ${outlierColumns.join(', ')} - may affect clustering`);
      }
      
      if (numericColumns.length > 10) {
        recommendations.push(`📉 Consider dimensionality reduction (PCA) - ${numericColumns.length} features detected`);
      }
    }

    // Generate comprehensive insights
    const insights = generateInsights({
      shape,
      numericColumns,
      objectColumns,
      missingValues,
      numericalSummary,
      objectSummary,
      correlation,
      validationChecks,
      goal
    });

    const isValid = validationChecks.hasData && 
                    validationChecks.hasColumns && 
                    validationChecks.noEmptyColumns &&
                    validationChecks.missingDataLevel === 'Acceptable';

    setEdaResults({
      shape,
      columns: headers,
      columnTypes,
      numericColumns,
      objectColumns,
      missingValues,
      uniqueValues,
      numericalSummary,
      objectSummary,
      correlation,
      validationChecks,
      isValid,
      goal,
      recommendations,
      insights
    });

    setIsProcessing(false);
    return { validationChecks, isValid, recommendations, insights };
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
        const text = event.target.result;
        const parsedData = parseCSV(text);
        setDataset(parsedData);
        
        setMessages(prev => [...prev, { 
          type: 'agent', 
          content: `✅ Dataset loaded successfully!\n\n📊 Dataset Info:\n- Rows: ${parsedData.rows.length}\n- Columns: ${parsedData.headers.length}\n- Column Names: ${parsedData.headers.join(', ')}\n\n🎯 What's your goal with this dataset?\n\nExamples:\n• "I want to predict sales"\n• "I want to cluster customers"\n• "Just explore and analyze the data"\n\nTell me your goal, and I'll tailor the analysis accordingly!` 
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
      
      let conclusionMsg = `✅ **Analysis Complete!**\n\n`;
      conclusionMsg += `${result.isValid ? '✅ Dataset Validation: PASSED' : '⚠️ Dataset Validation: ISSUES DETECTED'}\n\n`;
      
      if (result.recommendations.length > 0) {
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

    // Handle data queries
    if (edaResults) {
      let response = '';
      
      if (userMessage.toLowerCase().includes('insight') || userMessage.toLowerCase().includes('finding')) {
        if (edaResults.insights) {
          response = `🔍 **Comprehensive Insights:**\n\n`;
          
          if (edaResults.insights.dataQuality.length > 0) {
            response += `**📊 Data Quality:**\n${edaResults.insights.dataQuality.map(i => `• ${i}`).join('\n')}\n\n`;
          }
          
          if (edaResults.insights.featureInsights.length > 0) {
            response += `**🔬 Feature Analysis:**\n${edaResults.insights.featureInsights.map(i => `• ${i}`).join('\n')}\n\n`;
          }
          
          if (edaResults.insights.correlationInsights.length > 0) {
            response += `**🔗 Correlation Insights:**\n${edaResults.insights.correlationInsights.map(i => `• ${i}`).join('\n')}\n\n`;
          }
          
          if (edaResults.insights.distributionInsights.length > 0) {
            response += `**📈 Distribution Analysis:**\n${edaResults.insights.distributionInsights.map(i => `• ${i}`).join('\n')}\n\n`;
          }
          
          if (edaResults.insights.actionableRecommendations.length > 0) {
            response += `**💡 Actionable Recommendations:**\n${edaResults.insights.actionableRecommendations.map(i => `${i}`).join('\n')}`;
          }
        } else {
          response = 'Insights are being generated. Please wait for the analysis to complete.';
        }
      }
      else if (userMessage.toLowerCase().includes('variance') || userMessage.toLowerCase().includes('iqr')) {
        response = `📊 **Variance & IQR Analysis:**\n\n`;
        Object.entries(edaResults.numericalSummary).slice(0, 5).forEach(([col, stats]) => {
          response += `**${col}:**\n`;
          response += `  • Variance: ${stats.variance}\n`;
          response += `  • IQR: ${stats.iqr} (Q1: ${stats.q1}, Q3: ${stats.q3})\n`;
          response += `  • Range: ${stats.range} (${stats.min} to ${stats.max})\n`;
          response += `  • Outlier Bounds: [${stats.lowerBound}, ${stats.upperBound}]\n`;
          response += `  • Coefficient of Variation: ${stats.coefficientOfVariation}%\n\n`;
        });
      }
      else if (userMessage.toLowerCase().includes('unique') || userMessage.toLowerCase().includes('cardinality')) {
        response = `🔢 **Unique Values Analysis:**\n\n`;
        response += `**Numerical Features:**\n`;
        Object.entries(edaResults.numericalSummary).forEach(([col, stats]) => {
          const uniqueRatio = (stats.unique / stats.count * 100).toFixed(2);
          response += `  • ${col}: ${stats.unique} unique values (${uniqueRatio}% of total)\n`;
        });
        response += `\n**Categorical Features:**\n`;
        Object.entries(edaResults.objectSummary).forEach(([col, stats]) => {
          response += `  • ${col}: ${stats.unique} unique values (${stats.uniquePercentage}% of total)\n`;
        });
      }
      else if (userMessage.toLowerCase().includes('shape') || userMessage.toLowerCase().includes('size')) {
        response = `📊 **Dataset Shape:**\n- Rows: ${edaResults.shape.rows}\n- Columns: ${edaResults.shape.columns}\n- Total cells: ${edaResults.shape.rows * edaResults.shape.columns}`;
      } 
      else if (userMessage.toLowerCase().includes('column') || userMessage.toLowerCase().includes('feature')) {
        response = `📋 **Columns (${edaResults.columns.length}):**\n\n`;
        response += `**Numerical (${edaResults.numericColumns.length}):** ${edaResults.numericColumns.join(', ') || 'None'}\n\n`;
        response += `**Categorical (${edaResults.objectColumns.length}):** ${edaResults.objectColumns.join(', ') || 'None'}`;
      } 
      else if (userMessage.toLowerCase().includes('missing') || userMessage.toLowerCase().includes('null')) {
        const hasMissing = Object.values(edaResults.missingValues).some(m => m.count > 0);
        if (hasMissing) {
          response = `⚠️ **Missing Values Detected:**\n\n`;
          Object.entries(edaResults.missingValues)
            .filter(([_, v]) => v.count > 0)
            .forEach(([col, v]) => {
              response += `• ${col}: ${v.count} (${v.percentage}%) - ${v.severity} severity\n`;
            });
        } else {
          response = '✅ **No missing values found!** Your dataset is complete.';
        }
      } 
      else if (userMessage.toLowerCase().includes('outlier')) {
        const outlierCols = Object.entries(edaResults.numericalSummary)
          .filter(([_, stats]) => stats.outliers > 0);
        
        if (outlierCols.length > 0) {
          response = `🔍 **Outliers Detected:**\n\n`;
          outlierCols.forEach(([col, stats]) => {
            response += `• ${col}: ${stats.outliers} outliers (${stats.outliersPercentage}%)\n`;
          });
        } else {
          response = '✅ No significant outliers detected using IQR method.';
        }
      }
      else if (userMessage.toLowerCase().includes('correlation')) {
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
    for col in numerical_cols:
        print(f"\\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Variance: {df[col].var():.2f}")
        
        # Quartiles and IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        print(f"  Q1 (25th percentile): {Q1:.2f}")
        print(f"  Q3 (75th percentile): {Q3:.2f}")
        print(f"  IQR (Interquartile Range): {IQR:.2f}")
        print(f"  Lower Bound (Q1 - 1.5*IQR): {(Q1 - 1.5*IQR):.2f}")
        print(f"  Upper Bound (Q3 + 1.5*IQR): {(Q3 + 1.5*IQR):.2f}")
        
        # Range
        data_range = df[col].max() - df[col].min()
        print(f"  Range: {data_range:.2f}")
        
        # Unique values
        unique_count = df[col].nunique()
        unique_ratio = (unique_count / len(df)) * 100
        print(f"  Unique Values: {unique_count} ({unique_ratio:.2f}% of total)")
        
        # Skewness and Kurtosis
        print(f"  Skewness: {df[col].skew():.2f}")
        print(f"  Kurtosis: {df[col].kurtosis():.2f}")
        
        # Coefficient of Variation
        cv = (df[col].std() / df[col].mean()) * 100 if df[col].mean() != 0 else 0
        print(f"  Coefficient of Variation: {cv:.2f}%")

# Categorical columns summary
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if categorical_cols:
    print(f"\\nCategorical Columns ({len(categorical_cols)}):")
    for col in categorical_cols:
        print(f"\\n{col}:")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Most common value: {df[col].mode()[0]}")
        print(f"\\n  Top 5 values:")
        print(df[col].value_counts().head())

# ============================================
# STEP 5: OUTLIER DETECTION
# ============================================

print("\\n" + "="*50)
print("OUTLIER DETECTION (IQR Method)")
print("="*50)

outlier_summary = []

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / len(df)) * 100
    
    if outlier_count > 0) {
        outlier_summary.append({
            'Column': col,
            'Outliers': outlier_count,
            'Percentage': f"{outlier_pct:.2f}%",
            'Lower_Bound': f"{lower_bound:.2f}",
            'Upper_Bound': f"{upper_bound:.2f}"
        });
    }

if (outlier_summary.length > 0) {
    console.log("\\nOutliers detected:");
    // Note: original code is Python - please run server-side for Python outputs
} else {
    console.log("\\n✅ No significant outliers detected!");
}

export default ValidationAgenticAI;
