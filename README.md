# 🏦 Credit Risk Modeling - Machine Learning Project

![Credit Risk Analysis](https://images.unsplash.com/photo-1554224155-6726b3ff858f?w=1200&h=300&fit=crop)

> **Automated credit scoring system using Machine Learning to predict loan default risk**

## 📋 Project Overview

This project implements a complete machine learning pipeline for **Credit Risk Modeling** (Credit Scoring), helping financial institutions predict whether a loan applicant will default on their loan. The system uses advanced ML algorithms to analyze borrower characteristics and make accurate predictions, reducing manual effort and improving decision-making accuracy.

### Why This Matters

When banks lend money, they face the risk that borrowers won't repay (*Credit Risk*). Traditional manual analysis is:
- ⏱️ Time-consuming
- 💰 Expensive
- 🎯 Less accurate

**Machine Learning automates this process**, providing faster, more precise risk assessments.

---

## 🎯 Key Features

- ✅ **Modular Architecture**: Clean separation of concerns (preprocessing, models, main)
- ✅ **Advanced Data Preprocessing**: Missing value imputation, outlier detection (IQR), SMOTE balancing
- ✅ **Multiple ML Models**: Naive Bayes & Logistic Regression with comparative analysis
- ✅ **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- ✅ **Interactive Jupyter Notebook**: Step-by-step analysis with visualizations
- ✅ **Production-Ready Code**: Well-documented, modular, and maintainable

---

## 📊 Dataset

- **Source**: `credit_risk_dataset.csv`
- **Size**: 32,581 records × 12 features
- **Target Variable**: `loan_status` (0 = paid, 1 = defaulted)

### Features Include:
- **Demographic**: Age, income, employment length
- **Loan Details**: Amount, interest rate, purpose, grade
- **Credit History**: Previous defaults, credit history length

---

## 🏗️ Project Structure

```
Credit-Risk-Modeling-master/
│
├── 📊 credit_risk_dataset.csv              # Dataset (32K+ records)
├── 📓 Credit_Scoring_Master_Analysis.ipynb # Comprehensive Jupyter analysis
│
├── 🐍 Python Modules (Modular Architecture)
│   ├── main.py                             # Main orchestrator
│   ├── data_preprocessing.py               # Data cleaning & transformation
│   ├── models.py                           # ML model training & evaluation
│   └── plotly_config.py                    # Visualization configuration
│
├── 📦 Configuration
│   ├── requirements.txt                    # Python dependencies
│   └── .gitignore                          # Git exclusions
│
└── 📈 Output
    └── roc_curve_comparison.png            # Model performance visualization
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone or download this repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies

```
pandas              # Data manipulation
numpy               # Numerical computing
matplotlib          # Plotting
seaborn             # Statistical visualization
scikit-learn        # ML algorithms
imbalanced-learn    # SMOTE for class balancing
jupyter             # Interactive notebooks
ipykernel           # Jupyter kernel
plotly              # Interactive visualizations
nbformat            # Notebook formatting
```

---

## 💻 Usage

### Option 1: Run the Complete Pipeline (Python Script)

```bash
python main.py
```

**This will**:
1. Load and preprocess the dataset
2. Train Naive Bayes and Logistic Regression models
3. Evaluate both models
4. Generate ROC curve comparison (`roc_curve_comparison.png`)
5. Display performance metrics

### Option 2: Interactive Analysis (Jupyter Notebook)

```bash
jupyter notebook Credit_Scoring_Master_Analysis.ipynb
```

**The notebook includes**:
- 📊 Exploratory Data Analysis (EDA)
- 🔧 Step-by-step preprocessing
- 🤖 Model training with explanations
- 📈 Advanced visualizations
- 🎯 Threshold optimization
- 💡 Business insights

---

## 🧪 Pipeline Architecture

### 1. **Data Preprocessing** (`data_preprocessing.py`)

```python
from data_preprocessing import prepare_data

X_train, X_test, y_train, y_test = prepare_data()
```

**Steps:**
- ✅ Load CSV data
- ✅ Handle missing values (median/mode imputation)
- ✅ Remove outliers (IQR method)
- ✅ Encode categorical features (One-Hot Encoding)
- ✅ Normalize numerical features (StandardScaler)
- ✅ Balance classes with SMOTE (Synthetic Minority Oversampling)
- ✅ Split into train/test (80/20 stratified)

### 2. **Model Training** (`models.py`)

```python
from models import train_naive_bayes, train_logistic_regression

nb_model = train_naive_bayes(X_train, y_train)
lr_model = train_logistic_regression(X_train, y_train)
```

**Models Implemented:**
- **Naive Bayes**: Fast, probabilistic classifier
- **Logistic Regression**: Interpretable, industry-standard

### 3. **Evaluation** (`models.py`)

```python
from models import evaluate_model, get_roc_data

metrics = evaluate_model(model, X_test, y_test, "Model Name")
fpr, tpr, auc = get_roc_data(model, X_test, y_test)
```

**Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

---

## 📈 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **Naive Bayes** | ~0.85 | ~0.82 | ~0.88 | ~0.85 | ~0.90 |
| **Logistic Regression** | ~0.92 | ~0.90 | ~0.93 | ~0.91 | ~0.95 |

> **Winner**: Logistic Regression consistently outperforms across all metrics

### Visualizations

The project generates:
- 📊 Class distribution plots
- 🔥 Correlation heatmaps
- 📈 ROC curve comparisons
- 🎯 Threshold analysis charts
- 🎨 Feature importance plots

---

## 🧠 Key Concepts Explained

### What is SMOTE?
**SMOTE** (Synthetic Minority Oversampling Technique) creates synthetic samples of the minority class (loan defaults) to balance the dataset, preventing model bias toward the majority class.

### Why ROC-AUC?
The **ROC curve** shows the trade-off between catching defaults (True Positive Rate) and false alarms (False Positive Rate). **AUC** (Area Under Curve) summarizes performance: closer to 1.0 = better model.

### Threshold Optimization
The default 0.5 probability threshold may not be optimal. Lower thresholds catch more defaults but increase false alarms. The notebook explores this trade-off.

---

## 🛠️ Troubleshooting

### Plotly Rendering Error in Jupyter
If you see `ValueError: Mime type rendering requires nbformat>=4.2.0`:

**Solution**: Restart your Jupyter kernel (`Kernel` → `Restart Kernel`)

**Optional**: Add this to a cell after imports:
```python
import plotly.io as pio
pio.renderers.default = "notebook"
```

### Module Import Errors
Ensure you've activated the virtual environment and installed all dependencies:
```bash
pip install -r requirements.txt
```

---

## 📚 Learning Resources

This project is part of a series of Credit Risk Modeling tutorials:

### 🎥 Video Tutorials (YouTube)

1. **[Machine Learning for Credit Scoring in R](https://youtube.com/playlist?list=PLmJWMf9F8euTlCGNR9OQgUzvXmB3aCc3G)**
   - Complete course on building credit risk models in R
   - Covers GLM, Stepwise Regression, pROC library

2. **[Exploratory Data Analysis for Credit Risk (Python)](https://youtu.be/k8pifdQjtEA)**
   - In-depth EDA techniques
   - Discovering insights from data

3. **[Azure ML AutoML for Credit Scoring](https://youtu.be/OGqx0VOn1yg)**
   - Using Microsoft Azure's Automated ML
   - Model deployment as web service
   - Building Streamlit client application

4. **[Shiny Dashboard Integration](https://youtu.be/uU4kjctCFDw)**
   - Deploying Random Forest model in R Shiny
   - Interactive dashboard for predictions

📺 **YouTube Channel**: [J.A DATATECH CONSULTING](https://www.youtube.com/channel/UCpd56FfjlkKbkHlbgY6XE3w)

---

## 🤝 Contributing

Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests
- ⭐ Star this repository

---

## 📧 Contact

For questions or collaborations:
- 📺 YouTube: [J.A DATATECH CONSULTING](https://www.youtube.com/channel/UCpd56FfjlkKbkHlbgY6XE3w)
- 🔔 Subscribe, like, and share to support the channel!

---

## 📄 License

This project is open-source and available for educational purposes.

---

## 🎓 Skills You'll Learn

By working with this project, you'll develop expertise in:

- ✅ **Python Programming**: Functions, classes, modular design
- ✅ **Data Science**: Pandas, NumPy, data manipulation
- ✅ **Machine Learning**: Scikit-learn, model training, evaluation
- ✅ **Statistical Analysis**: EDA, correlation, distributions
- ✅ **Data Visualization**: Matplotlib, Seaborn, Plotly
- ✅ **Best Practices**: Virtual environments, Git, documentation

---

<div align="center">

**⭐ If you found this project helpful, please give it a star! ⭐**

Made with ❤️ for the Data Science Community

</div>
