# House Price Prediction

A Machine Learning regression project for predicting house prices from
property and location-related features.

## Project Overview

The goal of this project is to build and compare several regression
models and identify the model that provides the best house-price
predictions.

The project includes:

-   Data loading and basic exploration
-   Missing-value handling
-   Feature selection / removal of unnecessary columns
-   Exploratory Data Analysis (EDA)
-   Train/test split
-   Feature scaling
-   Training multiple regression models
-   Model comparison using regression metrics
-   Hyperparameter tuning with `GridSearchCV`
-   Feature importance analysis for tree-based models

## Dataset

The dataset used in this project is `House_Prices_ML_Project.csv`.

Main features include:

-   `Area`
-   `Rooms`
-   `Bathrooms`
-   `Floor`
-   `BuildingAge`
-   `Garage`
-   `Garden`
-   `Parking`
-   `SchoolDistance`
-   `HospitalDistance`
-   `CityCenterDistance`
-   `CrimeRate`
-   `PopulationDensity`
-   `HouseQuality`

Target:

-   `Price`

The `HouseID` column is treated as an identifier rather than a
predictive feature.

## Models

Several regression algorithms are compared, including:

-   Linear Regression
-   Ridge Regression
-   Lasso Regression
-   ElasticNet
-   SVR
-   KNN Regressor
-   Decision Tree Regressor
-   Random Forest Regressor
-   AdaBoost Regressor
-   Gradient Boosting / XGBoost where included in the final notebook

Tree-based models are trained without feature scaling, while models that
are sensitive to feature scale use standardized data.

## Evaluation

The models are evaluated using:

-   **R² Score** --- measures how much variance in house prices is
    explained by the model.
-   **MAE (Mean Absolute Error)** --- average absolute prediction error.
-   **MSE (Mean Squared Error)** --- average squared prediction error.

The models are compared in a results table to determine the strongest
baseline model.

## Hyperparameter Tuning

`GridSearchCV` is used to search for better hyperparameters for selected
tree-based models.

Example parameters explored for Random Forest include:

-   `n_estimators`
-   `max_depth`
-   `min_samples_split`

Cross-validation is used during the search to obtain a more reliable
estimate of model performance.

## Feature Importance

For the best tree-based model, feature importance is extracted and
associated with the original feature names.

This helps explain which property characteristics have the greatest
influence on the model's predictions.

## Project Structure

``` text
House-Price-Prediction/
├── house_price.ipynb
├── House_Prices_ML_Project.csv
├── README.md
├── requirements.txt
└── results/
    └── feature_importance.png
```

## Libraries

Main Python libraries used:

``` text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

## What I Learned

This project was used to practice a complete regression workflow:

1.  Understanding the dataset
2.  Preparing the data
3.  Separating features and target
4.  Splitting data into training and testing sets
5.  Understanding when scaling is necessary
6.  Comparing different regression algorithms
7.  Evaluating models with multiple metrics
8.  Using Grid Search for hyperparameter tuning
9.  Interpreting feature importance

## Author

This project was implemented as part of my Machine Learning portfolio
and learning journey.
