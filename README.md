#  Classification
## **Customer Classification Based on Clustering Results**

## Project Overview
This project focuses on building a **customer classification system** using supervised machine learning models. The classification labels are derived from a prior **clustering process**, enabling the transformation of unsupervised segmentation results into a predictive model.
The main goal is to predict customer segments (`Target`) based on transaction behavior, customer profile, and interaction patterns.

---

## Dataset
The dataset used in this project is **`data_clustering_inverse.csv`**, which is the inverse-transformed result of a previous clustering process.
It contains both **numerical and categorical features**, including:

* Transaction amount, duration, and account balance
* Customer age, occupation, and login behavior
* Transaction type, channel, and location
* Engineered time-based features from transaction timestamps
* Target label representing customer cluster

---

## Data Preprocessing

Several preprocessing steps were applied:

* Datetime feature extraction (hour, day of week, month)
* Feature engineering:

  * `TransactionHour`, `TransactionDayOfWeek`, `TransactionMonth`
  * `DaysSincePreviousTransaction`
* One-hot encoding for categorical features
* Missing value handling using median (numerical) and mode (categorical)
* Feature scaling using **StandardScaler** for distance-based models

---

## Models Implemented

The following classification models were trained and evaluated:

* **Decision Tree**
* **Random Forest**
* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**

To improve performance, **hyperparameter tuning** was performed on the Random Forest model using **GridSearchCV** with weighted F1-score as the evaluation metric.

---

## Model Evaluation

Each model was evaluated using:
* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-Score (weighted)

The tuned Random Forest model achieved the best overall performance and was selected as the final model.

---

## Saved Artifacts

The following models were saved using `joblib`:
* `decision_tree_model.h5`
* `explore_RandomForest_classification.h5`
* `explore_LogisticRegression_classification.h5`
* `explore_KNN_classification.h5`
* `tuning_classification.h5`

---

## Notes

This classification task was intentionally designed to **retain categorical features** rather than dropping them, allowing the models to learn richer behavioral patterns from customer data.

---

# README – Clustering
## **Customer Segmentation Using K-Means Clustering**

## Project Overview

This project aims to perform **customer segmentation** using **unsupervised learning**, specifically **K-Means Clustering**, on transaction data. The segmentation helps uncover customer behavior patterns that can be used for strategic business decisions such as targeted promotions and personalized services.

---

## Dataset

The dataset is a modified banking transaction dataset provided by Dicoding, consisting of **2,537 rows and 16 columns**, including:

* Transaction details (amount, duration, type)
* Customer demographics (age, gender, occupation)
* Location and channel information
* Login behavior and account balance

---

## Exploratory Data Analysis (EDA)

EDA was conducted to understand data distribution and relationships:

* Dataset overview and descriptive statistics
* Correlation matrix for numerical features
* Distribution histograms
* Advanced visualizations:
  1 Transaction type frequency
  2 Average transaction amount per channel
  3 Customer age distribution per transaction type

---

## Data Preprocessing

Key preprocessing steps include:
* Handling missing values and duplicates
* Dropping irrelevant ID columns
* Feature scaling using **MinMaxScaler**
* Label encoding for categorical features
* Outlier removal using IQR method
* Feature engineering:
  1 Age grouping (`AgeGroup`)
  2 Login frequency categorization (`LoginFreq`)

---

## Clustering Process
* **Elbow Method** using `KElbowVisualizer` to determine optimal cluster count
* K-Means clustering using `sklearn`
* Model evaluation using **Silhouette Score**
* Dimensionality reduction using **PCA** for visualization

---

## Cluster Interpretation

Each cluster was analyzed based on:

* Transaction amount
* Customer age
* Login behavior
* Account balance

The clusters represent different customer profiles such as:

* Moderate transaction customers
* High-value customers
* Stable financial behavior customers
* Savings-oriented customers

---

## Saved Files/Models
* `model_clustering.h5`
* `PCA_model_clustering.h5`
* `data_clustering.csv`
* `data_clustering_inverse.csv`

---

## Outcome

The clustering results provide actionable insights into customer behavior and were later used as labels in the classification project, forming an **end-to-end machine learning pipeline** from unsupervised to supervised learning.
