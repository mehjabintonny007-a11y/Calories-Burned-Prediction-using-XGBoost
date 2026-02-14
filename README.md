🏋️ Calories Burned Prediction using XGBoost
Project Overview

Calories burned during exercise depends on multiple physiological factors like age, weight, height, heart rate, exercise duration, and body temperature.
This project aims to predict the number of calories burned based on these features using XGBoost Regressor, a gradient boosting machine learning model ideal for complex interactions.

Dataset
Two CSV files are used:

exercise.csv – contains features like:
User_ID – unique user identifier
Gender – male/female
Age – in years
Height – in cm
Weight – in kg
Duration – duration of exercise in minutes
Heart_Rate – average heart rate during exercise
Body_Temp – body temperature during exercise

calories.csv – contains:

User_ID – unique user identifier
Calories – calories burned during exercise

The datasets are combined into one dataframe using User_ID to align the features with target values.

Data Preprocessing

Gender Encoding:
male → 0, female → 1 for numeric compatibility with XGBoost.

Missing Values:
Checked using isnull().sum() and handled if any.

Features & Target:
X = calories_data.drop(['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

Exploratory Data Analysis (EDA)

Numerical Features
Age, Height, Weight, Duration, Heart Rate, Body Temperature distributions visualized using matplotlib and seaborn.

Categorical Features
Gender distribution plotted using bar charts and seaborn countplot.

Correlation Analysis
Heatmap of feature correlations:
sns.heatmap(correlation, cbar=True, square=True, annot=True, fmt='.1f', annot_kws={'size':8, 'color':'k'}, cmap='Blues')
Positive and negative correlations with Calories are highlighted.

Model Training

Model Used: XGBoost Regressor

Parameters:

XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    random_state=2
)

Train/Test Split:

70% training data
30% testing data
Random state = 2 for reproducibility

Training:
regressor.fit(X_train, Y_train)

Evaluation
Metrics Used:
R² Score – goodness of fit
Mean Absolute Error (MAE) – average error in calories

Results:

R² Score (Training): train_data_accuracy
R² Score (Testing): test_data_accuracy
Mean Absolute Error: mae
A high R² on both training and testing sets indicates the model captures the relationship between features and calories burned well.

Visualization Samples

Histograms for Age, Height, Weight, Duration, Heart Rate, Body Temperature
Bar plot and countplot for Gender distribution
Heatmap showing correlations between features and Calories

Conclusion
XGBoost successfully predicts calories burned based on physiological and exercise data.
Feature importance shows Duration, Heart Rate, and Weight have significant impact on calorie expenditure.
This model can be integrated into fitness apps or wearable devices to estimate calories burned in real-time.

Future Work

Add more features like exercise type, intensity, or resting metabolic rate
Hyperparameter tuning using GridSearchCV for better performance
Deploy as a web app or mobile app for real-time predictions

How to Run
Clone the repository
Place exercise.csv and calories.csv in /content/ or project folder
Run the Python notebook:
python calories_prediction.py
Visualizations and model results will be generated automatically.
Technologies Used
Python 3.x
Pandas, NumPy for data handling
Matplotlib, Seaborn for visualization
Scikit-learn for train/test split and evaluation
XGBoost for regression modeling

Author
Mehjabin Tonny
