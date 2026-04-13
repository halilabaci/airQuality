# Models / Experimental Results / Comparison

Dataset used for the experiments: `pilot_clean_encoded.csv`.
Targets were created as PM2.5 forecasts for t+1, t+8, and t+24 hours using chronological ordering.
A chronological 80/20 train-test split was used to avoid leaking future observations into training.

## Models
The following regressors were evaluated: Linear Regression, Decision Tree Regressor, Random Forest Regressor, SVR, and Gradient Boosting Regressor.
MAE, RMSE, and R2 were reported for each horizon. Lower MAE/RMSE and higher R2 indicate better performance.

## Experimental Results
For t+1, the best baseline model was Linear Regression with RMSE=18.785, MAE=10.470, and R2=0.950.
After feature selection, the best model was Linear Regression with RMSE=18.858, MAE=10.480, and R2=0.950.
For t+8, the best baseline model was Gradient Boosting Regressor with RMSE=56.081, MAE=35.324, and R2=0.558.
After feature selection, the best model was Gradient Boosting Regressor with RMSE=58.376, MAE=36.819, and R2=0.521.
For t+24, the best baseline model was Linear Regression with RMSE=73.139, MAE=50.663, and R2=0.248.
After feature selection, the best model was Gradient Boosting Regressor with RMSE=73.043, MAE=51.217, and R2=0.250.

## Comparison
For t+1, feature selection worsened the best RMSE by 0.072 (Linear Regression -> Linear Regression).
For t+8, feature selection worsened the best RMSE by 2.295 (Gradient Boosting Regressor -> Gradient Boosting Regressor).
For t+24, feature selection improved the best RMSE by 0.095 (Linear Regression -> Gradient Boosting Regressor).

## Selected Features
Top selected features for t+1: PM2.5, PM10, CO, NO2, SO2.
Top selected features for t+8: PM2.5, PM10, CO, NO2, DEWP.
Top selected features for t+24: PM2.5, PM10, CO, NO2, DEWP.

## Best-Model Interpretation
The strongest selected-model result came from Linear Regression at t+1. The most influential features according to permutation importance were PM2.5 (88.592), PM10 (0.762), NO2 (0.621), DEWP (0.361), TEMP (0.212).

These paragraphs can be adapted directly into the report's Models, Experimental Results, and Comparison sections.