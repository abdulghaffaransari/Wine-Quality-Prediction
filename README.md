# Wine Quality Prediction Model

This project involves predicting the quality of red wine using ElasticNet, Ridge, and Lasso regression models. The script uses MLflow for tracking experiments, logging metrics, and saving models.

## Usage

### Data Preprocessing

1. **Load the dataset**: The dataset is loaded from `winequality-red.csv`.
2. **Feature and Target Separation**: Separate features and the target variable `quality`.
3. **Train-Test Split**: Divide the dataset into training and testing sets.

### Model Training

1. **Train Models**:
   - **ElasticNet**: Trained with specified `alpha` and `l1_ratio` parameters.
   - **Ridge Regression**: Trained with a specified `alpha` parameter.
   - **Lasso Regression**: Trained with a specified `alpha` parameter.

2. **Logging and Evaluation**:
   - Log model parameters and metrics using MLflow.
   - Evaluate models using RMSE, MAE, and R2 score.
   - Save the trained models using MLflow.

### MLflow Tracking

- **Tracking**: MLflow is used to track experiments, log model parameters, metrics, and artifacts.

## Scripts

- **`train_model.py`**: The main script that performs data loading, model training, evaluation, and MLflow logging.

### Script Parameters

You can adjust the following parameters when running the script:

- `--alpha` or `-a`: ElasticNet alpha value (default: 0.2)
- `--l1_ratio` or `-l1`: ElasticNet l1 ratio (default: 0.3)
- `--ridge_alpha` or `-ra`: Ridge alpha value (default: 1.0)
- `--lasso_alpha` or `-la`: Lasso alpha value (default: 1.0)


To run the script with default parameters:

### Results
Models are evaluated based on RMSE, MAE, and R2 score.
Models and evaluation metrics are logged in MLflow.

