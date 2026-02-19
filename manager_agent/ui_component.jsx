import React from 'react'

export default function MLConfig() {
  return (
    <div style={{fontFamily: 'Arial, sans-serif', padding: 18}}>
      <h2>Model Configuration</h2>

      <section style={{marginBottom: 24}}>
        <h3>1. Preprocessing & Feature Engineering</h3>
        <p>Select the target column and configure imputers, scaling and encoding.</p>
        <ul>
          <li>Target column: <em>select from dataset</em></li>
          <li>Numeric imputation: <strong>median</strong> / mean / knn</li>
          <li>Categorical imputation: most_frequent / constant</li>
          <li>Scaling: none / standard / minmax</li>
          <li>Encoding: one-hot / ordinal / target</li>
          <li>Feature selection: variance / select_k_best / PCA</li>
        </ul>
      </section>

      <section style={{marginBottom: 24}}>
        <h3>2. Model Creation, Training & Evaluation</h3>
        <p>Choose candidate algorithms, cross-validation and metrics to evaluate.</p>
        <ul>
          <li>Candidate models: logistic_regression, random_forest, xgboost, svm</li>
          <li>Hyperparameter search: none / grid / random / bayes</li>
          <li>CV folds: 3 / 5 / 10</li>
          <li>Metrics: accuracy, f1, roc_auc, precision, recall</li>
        </ul>
      </section>

      <section>
        <h3>3. Model Comparison & Selection</h3>
        <p>Compare models by a chosen metric and review feature importance and explainability.</p>
        <ul>
          <li>Compare by: accuracy / f1 / roc_auc</li>
          <li>Explainability: SHAP / LIME / none</li>
          <li>Deploy options: Docker / AWS SageMaker / Download artifact</li>
        </ul>
      </section>
    </div>
  )
}
