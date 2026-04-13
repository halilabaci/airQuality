# Models / Experimental Results / Comparison

Dataset used for the experiments: `air_quality_all_clean_encoded.csv`.
Targets were created as PM2.5 forecasts for t+1, t+8, and t+24 hours using chronological ordering.
A chronological 80/20 train-test split was used to avoid leaking future observations into training.

## Models
The following regressors were evaluated: Linear Regression, Decision Tree Regressor, Random Forest Regressor, SVR, and Gradient Boosting Regressor.
MAE, RMSE, and R2 were reported for each horizon. Lower MAE/RMSE and higher R2 indicate better performance.

## Experimental Results
For t+1, the best baseline model was Random Forest Regressor with RMSE=20.100, MAE=11.069, and R2=0.944.
After feature selection, the best model was Random Forest Regressor with RMSE=20.295, MAE=11.185, and R2=0.943.
For t+8, the best baseline model was Random Forest Regressor with RMSE=44.668, MAE=28.401, and R2=0.724.
After feature selection, the best model was Random Forest Regressor with RMSE=47.742, MAE=30.578, and R2=0.684.
For t+24, the best baseline model was Random Forest Regressor with RMSE=47.822, MAE=32.567, and R2=0.683.
After feature selection, the best model was Random Forest Regressor with RMSE=50.681, MAE=34.948, and R2=0.644.

## Comparison
For t+1, feature selection worsened the best RMSE by 0.195 (Random Forest Regressor -> Random Forest Regressor).
For t+8, feature selection worsened the best RMSE by 3.073 (Random Forest Regressor -> Random Forest Regressor).
For t+24, feature selection worsened the best RMSE by 2.859 (Random Forest Regressor -> Random Forest Regressor).

## Selected Features
Top selected features for t+1: PM2.5, PM10, CO, NO2, SO2.
Top selected features for t+8: PM2.5, PM10, CO, NO2, SO2.
Top selected features for t+24: PM2.5, PM10, CO, NO2, month.

## Best-Model Interpretation
The strongest selected-model result came from Random Forest Regressor at t+1. The most influential features according to permutation importance were PM2.5 (70.585), PM10 (16.170), DEWP (2.032), WSPM (1.952), NO2 (1.827).

These paragraphs can be adapted directly into the report's Models, Experimental Results, and Comparison sections.