import os
import sys
import pickle

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,roc_auc_score
)

from sklearn.model_selection import RandomizedSearchCV

from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:

            obj = pickle.load(file_obj)

        logging.info("Object loaded successfully")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


param = {
    "Logistic Regression": {
        "C": [0.1, 1, 10],
        "solver": ["liblinear"],
        "class_weight": ["balanced"]
    },

    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"]
    },

    "Random Forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"]
    },

    "Gradient Boosting": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7]
    },

    "AdaBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 1.0]
    }
}


def evaluate_model(
    x_train,
    y_train,
    x_test,
    y_test,
    models,
    param
):
    try:

        report = {}
        best_params = {}
        best_models = {}

        for model_name, model in models.items():

            logging.info(
                f"Starting hyperparameter tuning for {model_name}"
            )

            gs = RandomizedSearchCV(
                estimator=model,
                param_distributions=param[model_name],
                n_iter=10,
                cv=3,
                scoring="f1",
                random_state=42,
                n_jobs=-1
            )

            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_

            best_params[model_name] = gs.best_params_

            best_models[model_name] = best_model

            y_pred = best_model.predict(x_test)
            y_prob = best_model.predict_proba(x_test)[:, 1]
           
            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

            report[model_name] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": roc_auc
            }

            logging.info(
                f"{model_name} Best Params: {gs.best_params_}"
            )

            logging.info(
                f"{model_name} Precision: {precision:.4f}"
            )

            logging.info(
                f"{model_name} Recall: {recall:.4f}"
            )

            logging.info(
                f"{model_name} F1 Score: {f1:.4f}"
            )
            

            logging.info(
                f"{model_name} ROC AUC: {roc_auc:.4f}"
            )

        return report, best_params, best_models

    except Exception as e:
        raise CustomException(e, sys)