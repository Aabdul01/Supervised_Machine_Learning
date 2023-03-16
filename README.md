House Price Prediction Using Supervised Machine Learning
This repository contains code for predicting house prices using supervised machine learning. The dataset used for this project is housing_prices7.csv.

Data Preprocessing
We first loaded the dataset using pandas read_csv function. The target variable SalePrice was separated from the rest of the features using the pop method. The Id column was also removed from the feature matrix as it provides no predictive power.

We then used the make_column_transformer function from Scikit-learn to apply preprocessing pipelines to the different data types in the feature matrix. The preprocessing pipelines included imputation for missing values using the mean and constant strategies for numerical and categorical features, respectively. We also used ordinal encoding and one-hot encoding for ordinal and nominal categorical features, respectively.

Model Selection
We used the GridSearchCV function from Scikit-learn to perform hyperparameter tuning for our decision tree model. We defined a parameter grid for the DecisionTreeRegressor including the criterion, splitter, and minimum samples leaf parameters.

We also created a pipeline for our model using make_pipeline function. The pipeline included the preprocessing transformers and the DecisionTreeRegressor estimator.

Model Performance
We evaluated our model using various metrics such as mean absolute error (MAE), root mean squared error (RMSE), mean absolute percentage error (MAPE), and R-squared (R2) scores. We used the mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, and r2_score functions from Scikit-learn to compute these metrics.

Finally, we created a dataframe to compare the performance of our initial model and the tuned decision tree model.

Conclusion
This project demonstrates how to use supervised machine learning to predict house prices. The decision tree model achieved reasonable performance after hyperparameter tuning. This repository can serve as a starting point for further exploration and experimentation with other models and datasets.
