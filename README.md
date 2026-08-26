# Loan Approval Prediction

## 📌 Project Overview

This project uses **Machine Learning** to predict whether a loan application will be **Approved** or **Rejected**.

The model uses applicant financial and personal information such as income, loan amount, CIBIL score, assets, education, and employment status.

A **Logistic Regression** model is used for the classification task.

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Pickle
* Streamlit

---

## 📊 Dataset

The project uses the `loan_approval_dataset.csv` dataset.

The features used for prediction are:

* Number of dependents
* Education
* Self employed
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential assets value
* Commercial assets value
* Luxury assets value
* Bank asset value

The target variable is:

* `Approved` → 1
* `Rejected` → 0

---

## 🤖 Machine Learning Model

### Logistic Regression

Logistic Regression is used because this is a **binary classification problem**.

The model predicts one of two outcomes:

* **1 → Loan Approved**
* **0 → Loan Rejected**

---

## 🔄 Data Preprocessing

The following preprocessing steps are performed:

1. Load the dataset using Pandas.
2. Remove the `loan_id` column.
3. Clean column names using `strip()` and `lower()`.
4. Convert loan status into numerical values.
5. Split the dataset into training and testing sets.
6. Standardize numerical features using `StandardScaler`.
7. Encode categorical features using `OneHotEncoder`.
8. Combine numerical and categorical features.
9. Train the Logistic Regression model.

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

Example output:

```text
Accuracy: 0.9063231850117096
Precision: 0.9206642066420664
Recall: 0.9309701492537313
F1: 0.9257884972170687

Confusion Matrix:
[[275  43]
 [ 37 499]]
```

---

## 💾 Saved Models

After training, the following files are created using Pickle:

```text
loan_model.pkl
scaler.pkl
encoder.pkl
```

### `loan_model.pkl`

Contains the trained Logistic Regression model.

### `scaler.pkl`

Contains the fitted StandardScaler used for numerical features.

### `encoder.pkl`

Contains the fitted OneHotEncoder used for categorical features.

These files can be loaded later by the Streamlit application to make predictions without retraining the model.

---

## 📁 Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_dataset.csv
├── train.py
├── app.py
├── loan_model.pkl
├── scaler.pkl
├── encoder.pkl
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run the Project

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Install required libraries

Open the VS Code terminal and run:

```bash
pip install numpy pandas scikit-learn streamlit
```

### 3. Train the model

Run the training Python file:

```bash
python train.py
```

This will create:

```text
loan_model.pkl
scaler.pkl
encoder.pkl
```

### 4. Run the Streamlit application

After creating `app.py`, run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🎯 Project Goal

The goal of this project is to demonstrate how Machine Learning can be used to predict loan approval based on applicant information.

It also demonstrates the complete ML workflow:

**Data → Preprocessing → Training → Evaluation → Model Saving → Prediction**

---

## 👨‍💻 Author

Jagadeesh

BTech Computer Science & Engineering (AI/ML)
