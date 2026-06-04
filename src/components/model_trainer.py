import sys

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from src.logger import logging
from src.exception import CustomException
from src.utils import (
    save_object,
    evaluate_model,
    param
)

from src.components.model_trainer_config import (
    ModelTrainerConfig
)


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = (
            ModelTrainerConfig()
        )

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing data"
            )

            x_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            x_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression":
                    LogisticRegression(),

                "Decision Tree":
                    DecisionTreeClassifier(),

                "Random Forest":
                    RandomForestClassifier(),

                "Gradient Boosting":
                    GradientBoostingClassifier(),

                "AdaBoost":
                    AdaBoostClassifier()
            }

            (
                model_report,
                best_params,
                best_models
            ) = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=param
            )

            logging.info(
                f"Model Report: {model_report}"
            )

            logging.info(
                f"Best Parameters: {best_params}"
            )

            # Select best model based on ROC-AUC
            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["roc_auc"]
            )

            best_model = best_models[
                best_model_name
            ]

            best_model_metrics = model_report[
                best_model_name
            ]

            logging.info(
                f"Best Model: {best_model_name}"
            )

            logging.info(
                f"Precision: "
                f"{best_model_metrics['precision']:.4f}"
            )

            logging.info(
                f"Recall: "
                f"{best_model_metrics['recall']:.4f}"
            )

            logging.info(
                f"F1 Score: "
                f"{best_model_metrics['f1_score']:.4f}"
            )

            logging.info(
                f"ROC-AUC: "
                f"{best_model_metrics['roc_auc']:.4f}"
            )

            if (
                best_model_metrics["roc_auc"]
                < 0.80
            ):
                raise CustomException(
                    "No model achieved acceptable ROC-AUC",
                    sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            return {
                "model_name": best_model_name,
                "precision":
                    best_model_metrics["precision"],
                "recall":
                    best_model_metrics["recall"],
                "f1_score":
                    best_model_metrics["f1_score"],
                "roc_auc":
                    best_model_metrics["roc_auc"]
            }

        except Exception as e:
            raise CustomException(e, sys)