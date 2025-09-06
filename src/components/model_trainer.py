import os
import sys
import json
import pickle
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifact", "model.pkl")
    report_file_path: str = os.path.join("artifact", "model_report.json")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def _build_models_and_params(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Define candidate models and (small) param grids."""
        models = {
            "log_reg": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(),
            "gbdt": GradientBoostingClassifier(),
            "svc": SVC(class_weight="balanced"),
            "knn": KNeighborsClassifier(),
        }

        param_grids = {
            "log_reg": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l2"],  
                "solver": ["lbfgs", "liblinear"],
            },
            "random_forest": {
                "n_estimators": [100, 300],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "gbdt": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
            "svc": {
                "C": [0.5, 1.0, 2.0],
                "kernel": ["rbf", "linear"],
            },
            "knn": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
            },
        }
        return models, param_grids

    def _evaluate(self, y_true, y_pred) -> Dict[str, float]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }

    def initiate_model_trainer(self, train_arr: np.ndarray, test_arr: np.ndarray):
        """
        Expects the arrays returned by DataTransformation.initiate_data_transformation:
        final column = target, the rest = features.
        Trains 5 models with GridSearchCV, selects best by F1-macro, saves model & report.
        """
        try:
            logging.info("Splitting train/test arrays into X and y")
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

      
            if y_train.dtype.kind in {"f"}:
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)

            models, param_grids = self._build_models_and_params()
            results = {}

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)

            best_model_name = None
            best_model = None
            best_score = -1.0
            best_gs_summary = {}

            logging.info("Starting model selection with GridSearchCV")
            for name, model in models.items():
                logging.info(f"-> Tuning {name}")
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=param_grids[name],
                    scoring="f1_macro",  # robust for class imbalance
                    cv=5,
                    n_jobs=-1,
                    verbose=0,
                )
                gs.fit(X_train, y_train)

                y_pred_val = gs.predict(X_train)
                train_metrics = self._evaluate(y_train, y_pred_val)

                results[name] = {
                    "best_params": gs.best_params_,
                    "cv_best_score_f1_macro": float(gs.best_score_),
                    "train_metrics": train_metrics,
                }

                if gs.best_score_ > best_score:
                    best_score = gs.best_score_
                    best_model_name = name
                    best_model = gs.best_estimator_
                    best_gs_summary = {
                        "best_params": gs.best_params_,
                        "cv_best_score_f1_macro": float(gs.best_score_),
                    }

            logging.info(f"Best model: {best_model_name} with CV F1-macro={best_score:.4f}")

            best_model.fit(X_train, y_train)
            y_pred_test = best_model.predict(X_test)

            test_metrics = self._evaluate(y_test, y_pred_test)
            classif_rep = classification_report(y_test, y_pred_test, output_dict=True)

            with open(self.config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)
            logging.info(f"Saved best model to {self.config.trained_model_file_path}")

            report_payload = {
                "best_model": best_model_name,
                "best_model_cv": best_gs_summary,
                "test_metrics": test_metrics,
                "all_models": results,
                "classification_report": classif_rep,  # per-class precision/recall/f1
            }
            with open(self.config.report_file_path, "w", encoding="utf-8") as f:
                json.dump(report_payload, f, indent=2)

            logging.info(f"Saved report to {self.config.report_file_path}")
            return (
                self.config.trained_model_file_path,
                self.config.report_file_path,
                best_model_name,
                test_metrics,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation
    import os

    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        os.path.join("artifact", "train.csv"),
        os.path.join("artifact", "test.csv")
    )

    trainer = ModelTrainer()
    model_path, report_path, name, test_metrics = trainer.initiate_model_trainer(train_arr, test_arr)
    print("Best model:", name)
    print("Saved model at:", model_path)
    print("Saved report at:", report_path)
    print("Test metrics:", test_metrics)
