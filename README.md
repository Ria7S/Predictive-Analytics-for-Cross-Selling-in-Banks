# 💡 Predictive Analytics for Cross-Selling in Banking

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Machine Learning](https://img.shields.io/badge/Models-Decision%20Tree%20%7C%20Random%20Forest%20%7C%20GBM%20%7C%20XGBoost-orange)
![Clustering](https://img.shields.io/badge/Segmentation-KMeans-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📘 Project Overview  
This research applies **Predictive Analytics** to enhance **cross-selling strategies** in the banking sector, with a focus on **personal loans and related financial products**.  
Traditional marketing approaches often lack personalization and result in low response rates.  
By leveraging **machine learning models** and **customer segmentation**, this project identifies the most responsive customer segments, improving targeting accuracy and marketing efficiency.

---

## 🎯 Problem Statement  
> How can predictive analytics be used to optimize cross-selling strategies for personal loans and other bank products?

Banks face challenges in identifying the right customers for cross-selling due to heterogeneous customer behaviors and limited targeting efficiency.  
This project aims to create a **data-driven predictive model** that can recommend **personalized cross-sell offers** by analyzing customer demographics, transaction behavior, and product holdings.

---

## 🧠 Objectives  
- Evaluate and improve existing **cross-selling strategies** using predictive analytics.  
- Identify **key customer attributes** influencing loan or product acceptance.  
- Develop a **predictive model** that delivers response probabilities for each customer.  
- Use **clustering and segmentation** to recommend tailored financial products.

---

## 🧩 Methodology  

### 1️⃣ Data Collection  
- Dataset from a **bank’s CRM and financial systems (Kaggle)**.  
- Data includes **demographic**, **transactional**, and **product holding** attributes.  

### 2️⃣ Data Preprocessing  
- Removed irrelevant identifiers (e.g., `CUST_ID`), handled missing values, and capped outliers using **1st and 99th percentiles**.  
- Performed encoding for categorical variables using **dummy features**.  
- Addressed class imbalance using **ADASYN**, which generated synthetic samples for underrepresented responders.  
- Split the dataset into **70% training** and **30% testing** sets.

### 3️⃣ Exploratory Data Analysis (EDA)  
- Customer base consisted of **~12.5% responders** before modeling.  
- Gender, age, and balance distributions were analyzed; males and middle-aged groups showed slightly higher response rates.  
- Strong correlations observed between **balance, holding period, and credit score** with customer response.  
- Detected **non-linear feature relationships**, indicating **tree-based models** as suitable algorithms.

---

## ⚙️ Machine Learning Models  
Applied and compared the following models:
- 🌳 **Decision Tree** – Captures feature-based decision rules.  
- 🌲 **Random Forest** – Ensemble of trees to improve accuracy and reduce overfitting.  
- 🚀 **Gradient Boosting Machine (GBM)** – Sequentially minimizes errors by improving weak learners.  
- ⚡ **XGBoost** – Optimized gradient boosting framework providing the **highest predictive accuracy**.

### **Performance Summary (Test Data)**  
| Model | Accuracy | Precision | Recall | F1 Score |
|--------|-----------|------------|----------|------------|
| Decision Tree | 74.1% | 0.75 | 0.7 | 0.73 |
| Random Forest | 79% | 0.79 | 0.77 | 0.78 |
| Gradient Boost | 93.6% | 0.94 | 0.93 | 0.94 |
| **XGBoost** | **97.7%** | **0.98** | **0.97** | **0.97** |

✅ **XGBoost outperformed all models**, offering the most balanced performance across all metrics.

---

## 📉 Lorenz Curve & KS Analysis  
- Lorenz curve analysis revealed that **XGBoost** and **GBM** models achieved strong discriminatory power between responders and non-responders.  
- The **top 30%** of customers captured **~70% of total positive responses**, confirming high targeting efficiency.

---

## 🔍 Customer Segmentation (K-Means Clustering)  

Performed **K-Means clustering** on the cleaned, resampled dataset to group customers based on demographic and financial attributes.

| Cluster | Characteristics | Conversion Rate | Insights |
|----------|-----------------|-----------------|-----------|
| **Cluster 0** | Mid-age, moderate balance, high branch & cheque transactions | **59%** | Active customers ideal for cross-selling loans or investment products |
| **Cluster 1** | Young, highest balance, low transaction activity | **32%** | Low engagement; suitable for investment or saving schemes |
| **Cluster 2** | Long-term, moderate balance, high credit score | **60%** | Loyal customers; ideal for personalized offers & credit cards |

✅ Targeting **Clusters 0 & 2** can significantly improve cross-sell success.

---

## 📊 Key Findings  
- **XGBoost** achieved the best predictive accuracy (97%) and generalization power.  
- **ADASYN** successfully balanced the dataset, improving minority-class predictions.  
- **K-Means clustering** uncovered actionable customer segments with **up to 60% conversion rate**.  
- Implementing a **decile-based targeting strategy** improved marketing focus by **~40%**.  

---

## 🚀 Future Scope  
- Integrate **real-time retraining** and feedback loops to keep predictions updated.  
- Include **behavioral and sentiment features** for richer profiling.  
- Automate **segment-driven marketing campaigns** for personalized targeting.  
- Extend analysis to other financial products (credit cards, insurance, investments).

---

## 💻 Tech Stack  
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Plotly  
- **Techniques:** Predictive Modeling, ADASYN, Feature Engineering, K-Means Clustering, Lorenz Curve Analysis  

⭐ *If you found this project insightful, don’t forget to star the repository!* ⭐
