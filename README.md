# 🚀 Predictive Maintenance System using Machine Learning

## 📌 Project Overview

Predictive Maintenance is a machine learning application designed to predict machine failures before they occur. The system analyzes operational parameters such as air temperature, process temperature, rotational speed, torque, and tool wear to estimate the likelihood of machine failure.

The project follows an end-to-end Machine Learning lifecycle including data ingestion, data preprocessing, model training, model evaluation, model deployment, and a user-friendly Flask web application.

---

## 🎯 Problem Statement

Unexpected machine failures can lead to:

* Production downtime
* Increased maintenance costs
* Reduced operational efficiency
* Equipment damage

This project aims to predict machine failures proactively, enabling preventive maintenance and reducing operational risks.

---

## 📊 Dataset

**Dataset:** AI4I 2020 Predictive Maintenance Dataset

### Features

| Feature                 | Description                    |
| ----------------------- | ------------------------------ |
| Type                    | Machine Quality Type (L, M, H) |
| Air Temperature [K]     | Ambient air temperature        |
| Process Temperature [K] | Machine process temperature    |
| Rotational Speed [rpm]  | Machine rotational speed       |
| Torque [Nm]             | Applied torque                 |
| Tool Wear [min]         | Tool wear duration             |

### Target Variable

| Target                                        |
| --------------------------------------------- |
| Machine Failure (0 = No Failure, 1 = Failure) |

---

## 🏗️ Project Architecture

```text
Raw Dataset
      │
      ▼
Data Ingestion
      │
      ▼
Data Transformation
      │
      ▼
Preprocessor.pkl
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Best Model Selection
      │
      ▼
Model.pkl
      │
      ▼
Prediction Pipeline
      │
      ▼
Flask Web Application
```

---

## 🤖 Machine Learning Models Evaluated

The following classification algorithms were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* AdaBoost Classifier
* Gradient Boosting Classifier

---

## 🏆 Best Performing Model

### Gradient Boosting Classifier

| Metric   | Score  |
| -------- | ------ |
| ROC-AUC  | 96.88% |
| F1 Score | 70.27% |

Gradient Boosting achieved the highest overall performance and was selected as the final production model.

---

## 💻 Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-Learn
* Pandas
* NumPy

### Web Framework

* Flask

### Frontend

* HTML5
* CSS3
* Bootstrap 5

### Deployment

* AWS Elastic Beanstalk

### Version Control

* Git
* GitHub

---

## ✨ Features

### Machine Failure Prediction

Predict whether a machine is likely to fail.

### Failure Probability

Displays the probability of machine failure.

### Machine Health Score

Provides an overall machine health percentage.

### Risk Assessment

Classifies risk levels as:

* LOW
* MEDIUM
* HIGH

### Prediction Confidence

Shows model confidence in predictions.

### Recommendation Engine

Generates maintenance recommendations based on failure risk.

### Input Validation

Validates user inputs before prediction.

### Interactive Dashboard

Modern and responsive UI built using Bootstrap.

---

## 📷 Application Screenshots

### Dashboard

(Add screenshot here)

### Low Risk Prediction

(Add screenshot here)

### High Risk Prediction

(Add screenshot here)

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/tanveer123456/Predictive_Maintenance.git
```

### Navigate to Project

```bash
cd Predictive_Maintenance
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
python application.py
```

### Open Browser

```text
http://localhost:5000
```

---

## 📂 Project Structure

```text
Predictive_Maintenance/

├── artifacts/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── utils.py
│   ├── logger.py
│   └── exception.py
│
├── templates/
│   └── home.html
│
├── application.py
├── requirements.txt
├── README.md
└── setup.py
```

---

## 🔮 Future Improvements

* Docker Containerization
* CI/CD Pipeline
* Real-Time Monitoring Dashboard
* Cloud-Based Predictions
* Model Monitoring
* Feature Drift Detection
* Explainable AI (SHAP)

---

## 👨‍💻 Author

**Tanveer Chougule**

Machine Learning Enthusiast | Data Science Learner

GitHub:
https://github.com/tanveer123456

---

## ⭐ If you found this project useful, consider giving it a star.
