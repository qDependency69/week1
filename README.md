# week1

# 🚢 Titanic Survival Prediction

A machine learning project to predict passenger survival on the Titanic using binary classification.

## 🛠 Tech Stack
- **Data & Viz:** Pandas, NumPy, Matplotlib, Seaborn
- **ML Models:** Scikit-Learn, XGBoost, CatBoost
- **Explainability:** SHAP

## 🔄 Project Pipeline

1. **Preprocessing**  
   - Dropped `Name`, `Ticket`, `Cabin` (high cardinality/missing values).  
   - Filled `Age` with **median** (robust to outliers) and `Embarked` with **mode**.  
   - Applied One-Hot Encoding with `drop_first=True` to avoid the dummy variable trap.
2. **EDA**  
   - Visualized survival rates by gender, class, age, and fare.  
   - Plotted a correlation heatmap and 3D feature relationships.
3. **Modeling**  
   - Trained 6 models: Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, KNN.  
   - Applied `StandardScaler` *only* to linear/KNN models (tree-based models are scale-invariant).  
   - **Best Accuracy:** ~82% (Random Forest, XGBoost, CatBoost).
4. **Interpretability (SHAP)**  
   - Confirmed that `Sex`, `Fare`, and `Pclass` are the top predictors. SHAP values show exactly how each feature pushes the prediction toward survival (1) or death (0).

## 🚀 How to Run
1. Open the notebook in Google Colab or Jupyter.
2. Run the first cell to install dependencies:  
   `!pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost shap`
3. Upload `train.csv` to the working directory.
4. Execute all cells sequentially (`Shift + Enter`).

## 💡 Key Technical Decisions
- **Why median for Age?** Mean is skewed by outliers; median represents the typical passenger better.
- **Why `transform` (not `fit_transform`) on test data?** Prevents **data leakage**. The scaler must only learn parameters from the training set.
- **Why no scaling for trees?** Decision trees split on feature thresholds (e.g., `Age > 30`), so absolute feature scales do not affect their logic.

*Dataset: [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)*