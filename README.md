# 🚢 Titanic Survival Prediction using MLP (Neural Network)

## 📌 Overview

This project implements a **Multi-Layer Perceptron (MLP) Neural Network** to predict passenger survival on the Titanic dataset.

The model is optimized using:

- ✅ Stratified Train-Test Split  
- ✅ Standard Feature Scaling  
- ✅ Early Stopping  
- ✅ Hyperparameter Tuning (GridSearchCV)  
- ✅ ROC-AUC Optimization  
- ✅ Confusion Matrix Visualization  

---

## 📊 Dataset

Dataset used: Modified Titanic dataset  

Target Variable:
- `Survived`  
  - 0 = Died  
  - 1 = Survived  

### Features Used:

- Age  
- Fare  
- Sex  
- SibSp  
- Parch  
- Pclass  
- Embarked  

Irrelevant columns (such as `zero` columns and `Passengerid`) were removed during preprocessing.

---

## ⚙️ Project Workflow

### 1️⃣ Data Cleaning
- Removed duplicate rows
- Filled missing values in `Embarked`
- Dropped irrelevant columns
- Renamed incorrect target column

### 2️⃣ Preprocessing
- Feature scaling using `StandardScaler`
- Stratified train-test split (80/20)

### 3️⃣ Model Architecture

MLPClassifier with:
- Hidden Layers: Tuned (e.g., 128-64, 64-32, etc.)
- Activation Functions: ReLU / Tanh
- L2 Regularization (alpha)
- Early Stopping enabled
- Max Iterations: 1500

### 4️⃣ Hyperparameter Tuning

GridSearchCV with:
- 5-Fold Stratified Cross Validation
- ROC-AUC as scoring metric

---

## 📈 Model Evaluation

Metrics Used:

- Accuracy
- ROC-AUC Score
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 🎯 Expected Performance

- Accuracy: ~85%+
- ROC-AUC: ~0.88+

---

## 📊 Visualizations

- Confusion Matrix (Seaborn Heatmap)
- ROC Curve

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/Titanic-MLP-Survival-Prediction.git
cd Titanic-MLP-Survival-Prediction

pip install -r requirements.txt
jupyter notebook
```

Open the notebook:

```
MLP_Classifier_Titanic_survivor.ipynb
```
---

## 🔥 Future Improvements

- Add SHAP Explainability
- Try Ensemble (MLP + XGBoost)
- Add Polynomial Features
- Deploy using Streamlit
- Create Kaggle Submission Pipeline

---

## 👨‍💻 Author

**Devendra Kushwah**  
Machine Learning & AI Enthusiast
