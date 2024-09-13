import os
import mlflow
import argparse
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Ridge, Lasso
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_data():
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red"
    url = "winequality-red.csv"
    try:
        df = pd.read_csv(url, sep=",")
        return df
    except Exception as e:
        raise e


def eval_func(actual, pred):
    rmse = mean_squared_error(actual, pred, squared=False)  # root mean squared error
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_model(model_name, model, X_train, X_test, y_train, y_test, params):
    """Train the given model, log metrics, and save the model."""
    with mlflow.start_run():
        # Log model parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Fit and predict
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # Evaluate the model
        rmse, mae, r2 = eval_func(y_test, pred)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log the trained model
        input_example = X_test.iloc[:1].to_dict(orient='records')
        mlflow.sklearn.log_model(model, model_name, input_example=input_example)


def main(alpha, l1_ratio, ridge_alpha, lasso_alpha):
    df = load_data()
    TARGET = "quality"
    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # Split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)
    
    # Set the MLflow experiment
    mlflow.set_experiment("Wine-Quality-Prediction")

    # Model 1: ElasticNet
    elastic_net_params = {"alpha": alpha, "l1_ratio": l1_ratio}
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=6)
    train_model("ElasticNet", elastic_net, X_train, X_test, y_train, y_test, elastic_net_params)

    # Model 2: Ridge Regression
    ridge_params = {"alpha": ridge_alpha}
    ridge = Ridge(alpha=ridge_alpha, random_state=6)
    train_model("Ridge", ridge, X_train, X_test, y_train, y_test, ridge_params)

    # Model 3: Lasso Regression
    lasso_params = {"alpha": lasso_alpha}
    lasso = Lasso(alpha=lasso_alpha, random_state=6)
    train_model("Lasso", lasso, X_train, X_test, y_train, y_test, lasso_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", "-a", type=float, default=0.2, help="ElasticNet alpha value")
    parser.add_argument("--l1_ratio", "-l1", type=float, default=0.3, help="ElasticNet l1 ratio")
    parser.add_argument("--ridge_alpha", "-ra", type=float, default=1.0, help="Ridge alpha value")
    parser.add_argument("--lasso_alpha", "-la", type=float, default=1.0, help="Lasso alpha value")
    parsed_args = parser.parse_args()

    main(parsed_args.alpha, parsed_args.l1_ratio, parsed_args.ridge_alpha, parsed_args.lasso_alpha)
