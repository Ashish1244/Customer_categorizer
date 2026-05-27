# Customer_categorizer

Customer Personality Analysis & Segmentation Report
Project Overview
This project focuses on Customer Personality Analysis using machine learning, clustering, feature engineering, and classification techniques. The goal is to help businesses better understand customer behavior, purchasing patterns, and campaign responses in order to improve marketing strategies and customer targeting.
The project combines:
Exploratory Data Analysis (EDA)
Data Cleaning & Preprocessing
Feature Engineering
Customer Segmentation using Clustering
Classification Modeling
Deployment using Streamlit

1. Business Problem
Customer Personality Analysis helps organizations understand their customers in depth by analyzing:
Spending behavior
Purchase channels
Campaign participation
Family status
Income patterns
Product preferences
The objective is to:
Segment customers into meaningful groups
Predict customer clusters/personality types
Improve targeted marketing campaigns
Increase customer engagement and sales conversion

2. Dataset Information
The dataset contains customer demographic and behavioral information.
Dataset Characteristics
Total Records: 2240
Total Features: 29
Dataset Type: Tabular Marketing Dataset
Important Features
Demographic Features
Year_Birth
Education
Marital_Status
Income
Kidhome
Teenhome
Spending Features
MntWines
MntFruits
MntMeatProducts
MntFishProducts
MntSweetProducts
MntGoldProds
Purchase Features
NumWebPurchases
NumCatalogPurchases
NumStorePurchases
NumDealsPurchases
NumWebVisitsMonth
Campaign Features
AcceptedCmp1
AcceptedCmp2
AcceptedCmp3
AcceptedCmp4
AcceptedCmp5
Response

3. Exploratory Data Analysis (EDA)
The EDA notebook focuses on understanding data patterns, distributions, relationships, and anomalies.
Major EDA Steps
3.1 Data Understanding
Checked dataset dimensions
Analyzed feature types
Verified missing values
Explored duplicate records
3.2 Statistical Analysis
Performed:
Mean analysis
Median analysis
Standard deviation analysis
Distribution analysis
3.3 Visualizations Used
Histograms
Count plots
Correlation heatmaps
Boxplots
Pairplots
Distribution plots
3.4 Key Insights from EDA
Income Distribution
Income is highly varied among customers.
High-income customers generally spend more on premium products.
Product Preferences
Wine and meat products showed the highest spending.
Fruits and sweets had comparatively lower spending.
Purchase Behavior
Customers preferred store purchases more than web purchases.
Frequent web visitors were not always high spenders.
Campaign Analysis
Only a small percentage of customers accepted campaigns.
Previous campaign acceptance influenced future responses.
Family Structure
Customers with children had different spending patterns compared to customers without children.

4. Data Preprocessing & Feature Engineering
The second notebook focused on preparing the dataset for clustering and classification.
4.1 Data Cleaning
Missing Value Handling
Missing values in Income were identified and treated.
Duplicate Removal
Duplicate records were checked and removed if necessary.
Data Type Conversion
Date columns were converted into datetime format.

4.2 Feature Engineering
Several new features were created to improve model performance.
Engineered Features
Age
Calculated from Year_Birth:
Age = 2022 - Year_Birth
Children
Combined:
Kidhome + Teenhome
Parental Status
Binary feature:
1 → Has children
0 → No children
Total Spending
Combined spending across all products:
Wines
Fruits
Meat
Fish
Sweets
Gold Products
Total Promo
Combined accepted campaigns:
AcceptedCmp1 to AcceptedCmp5
Days as Customer
Calculated customer tenure using enrollment date.

4.3 Encoding
Education Encoding
Education levels were numerically encoded.
Marital Status Encoding
Marital categories were simplified into grouped classes.

5. Customer Segmentation Using Clustering
Agglomerative Clustering was used for customer segmentation.
Why Clustering?
Clustering helps:
Identify hidden customer groups
Discover similar customer behaviors
Enable personalized marketing
Improve recommendation systems
Clustering Technique Used
Agglomerative Clustering
This hierarchical clustering method groups customers based on similarity.
Benefits
Handles complex relationships
Suitable for customer segmentation
Produces meaningful clusters

6. Classification Modeling
After clustering customers, classification models were trained to predict customer clusters.
Objective
Predict which customer cluster a new customer belongs to.
Models Evaluated
Several machine learning models were compared:
Logistic Regression
Random Forest
AdaBoost
Gradient Boosting
XGBoost
LightGBM
CatBoost

6.1 Model Selection
GridSearchCV was used for:
Hyperparameter tuning
Model optimization
Performance improvement
Evaluation Metrics
Models were evaluated using:
Accuracy
Precision
Recall
F1-score

6.2 Best Performing Model
CatBoost Classifier
CatBoost was selected as the best-performing model.
Why CatBoost?
Handles categorical data effectively
Strong performance on tabular datasets
Reduces overfitting
Requires minimal preprocessing
The trained model was saved as:
catboost_model.pkl

7. Streamlit Deployment Application
A Streamlit web application was created for real-time customer cluster prediction.
Application Features
User Inputs
The application allows users to enter:
Customer demographics
Spending behavior
Purchase patterns
Campaign participation
Backend Processing
The app:
Performs preprocessing
Creates engineered features
Loads the trained CatBoost model
Predicts the customer cluster
Prediction Output
The app displays:
Predicted customer personality cluster

8. Technical Architecture
Technologies Used
Programming Language
Python
Libraries
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
CatBoost
Joblib
Streamlit
Deployment Tools
Streamlit

9. Workflow of the Project
End-to-End Pipeline
Data Collection
Data Cleaning
Exploratory Data Analysis
Feature Engineering
Clustering
Classification
Hyperparameter Tuning
Model Saving
Streamlit Deployment
Customer Prediction

10. Business Impact
This solution can help businesses:
Marketing Optimization
Run personalized campaigns
Target the right audience
Increase campaign conversion rates
Customer Understanding
Identify high-value customers
Detect low-engagement users
Understand buying behavior
Revenue Growth
Improve product recommendations
Increase customer retention
Enhance customer satisfaction

11. Strengths of the Project
Strong Feature Engineering
Custom features significantly improved customer understanding.
Complete ML Pipeline
The project includes:
EDA
Clustering
Classification
Deployment
Real-World Use Case
The project solves an actual business problem using data science.
Deployment Ready
The Streamlit application makes the solution interactive and production-oriented.

12. Limitations
Dataset Size
The dataset contains only 2240 records, which may limit generalization.
Static Dataset
The model is trained on historical data and may require retraining with new data.
Cluster Interpretability
Clusters may require additional business interpretation for production usage.
13. Future Improvements
Possible future enhancements:
Deep Learning Models
Experiment with neural networks for better classification.
Real-Time Data Integration
Integrate live customer data.
Advanced Segmentation
Use advanced clustering methods such as:
DBSCAN
Gaussian Mixture Models
K-Means Optimization
Explainable AI
Integrate SHAP or LIME for model explainability.
Cloud Deployment
Deploy using:
AWS
Azure
GCP
Docker

14. Conclusion
This project successfully demonstrates an end-to-end Customer Personality Analysis system using machine learning.
The workflow includes:
Data preprocessing
Customer segmentation
Feature engineering
Predictive modeling
Streamlit deployment
The CatBoost classifier effectively predicts customer clusters, enabling businesses to better understand customers and improve marketing strategies.
The project showcases practical data science skills including:
Data analysis
Machine learning
Clustering
Classification
Model deployment
Business understanding
Overall, this project is a strong example of applying data science techniques to solve real-world business problems.
