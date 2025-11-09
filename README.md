# 🛒 BigMart Sales Prediction

This project aims to predict product sales across different BigMart outlets using machine learning models. It involves extensive data preprocessing, feature analysis, and model evaluation to determine the best algorithm for accurate sales forecasting.

---

This project was completed as part of the **CSE422: Artificial Intelligence** course during the **Fall 2024** semester.

---

## 📂 Project Overview

BigMart, a large retail company, needs a data-driven way to forecast product sales for better inventory and marketing decisions.  
This project uses historical sales data to predict future sales using various regression models after thorough data preprocessing and feature selection.

---

**Contributors:** 

[MD. Shafiur Rahman](https://github.com/ShafiurShuvo) | [Md. Nafizur Rahman Bhuiya](https://github.com/MdNafizurRahmanBhuiya)

---

## 🧠 Objectives

- Analyze and preprocess the BigMart sales dataset  
- Handle missing and categorical data  
- Perform feature scaling and correlation analysis  
- Train multiple regression models  
- Compare performance metrics (R², RMSE, MSE, MAE)  
- Select the best-performing model for sales prediction  

---

## 📊 Dataset Information

**Dataset Name:** BigMart Sales Prediction  
**Source:** [Kaggle - BigMart Sales Prediction Dataset](https://www.kaggle.com/datasets/shivan118/big-mart-sales-prediction-datasets)  
**Reference:** BigMart Sales Prediction Dataset. (2018). Kaggle.  
**License:** Used for educational and research purposes only.

**Note:**  
A copy of the dataset has been uploaded to this repository for ease of execution and reproducibility. You can find the original owner in the Kaggle link

**Columns (12 features):**
- `Item_Identifier`  
- `Item_Weight`  
- `Item_Fat_Content`  
- `Item_Visibility`  
- `Item_Type`  
- `Item_MRP`  
- `Outlet_Identifier`  
- `Outlet_Establishment_Year`  
- `Outlet_Size`  
- `Outlet_Location_Type`  
- `Outlet_Type`  
- `Item_Outlet_Sales` *(Target)*  

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Handled missing values in `Item_Weight` and `Outlet_Size`
   - Encoded categorical features using Label and One-Hot Encoding
   - Removed unique columns that didn’t add predictive value
   - Standardized and scaled data using Min-Max scaling

2. **Feature Analysis**
   - Conducted correlation heatmap analysis
   - Removed low-impact features (e.g., `Item_Visibility`)

3. **Dataset Splitting**
   - 70% training data  
   - 30% testing data  

4. **Model Training**
   - Linear Regression  
   - Random Forest Regressor  
   - Decision Tree Regressor  
   - XGBoost Regressor  

5. **Evaluation Metrics**
   - R² Score  
   - RMSE (Root Mean Squared Error)  
   - MSE (Mean Squared Error)  
   - MAE (Mean Absolute Error)

---

## 📈 Model Comparison

| Model | R² Score | RMSE | MSE | MAE |
|:------|:----------|:------|:------|:------|
| **Linear Regression** | **0.5681** | **0.0843** | **0.0071** | **0.0620** |
| Random Forest | 0.5189 | 0.0889 | 0.0079 | 0.0620 |
| Decision Tree | 0.1403 | 0.1189 | 0.0141 | 0.0827 |
| XGBoost | 0.5039 | 0.0903 | 0.0082 | 0.0626 |

**✅ Best Performing Model:** Linear Regression

---

## 🧩 Results Summary

- **Linear Regression** performed best with the highest R² and lowest error metrics.  
- **Random Forest** and **XGBoost** showed competitive results but were slightly less accurate.  
- **Decision Tree** suffered from overfitting and lower generalization ability.  

---

## 🧾 Files in Repository

- `CSE422_Project1.ipynb` — Main Jupyter Notebook containing full code and outputs  
- `BigMart_Sales_Prediction_Report.pdf` — Detailed project report  
- `bigmart_sales.csv` — Dataset (from Kaggle, included for direct execution)

---

## 💡 Conclusion

Linear Regression was the most efficient and interpretable model for this dataset, achieving the best balance of accuracy and simplicity. With additional data and hyperparameter tuning, ensemble methods like Random Forest and XGBoost may yield even better performance.

---

## 🧰 Tools & Libraries Used

- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Scikit-learn  
- XGBoost  
- Jupyter Notebook
