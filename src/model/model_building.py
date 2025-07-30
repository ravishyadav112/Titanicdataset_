import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import yaml

n_estimators = yaml.safe_load(open("params.yaml" , 'r'))['model_building']['n_estimators']

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading featured data...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def scale_data(X: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    logger.info("Scaling data...")
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        test_scaled = scaler.transform(test)
        return X_scaled, test_scaled, scaler
    except Exception as e:
        logger.error(f"Error scaling data: {e}")
        raise


def train_model(X: np.ndarray, y: pd.Series) -> RandomForestClassifier:
    logger.info("Training model...")
    try:
        model = RandomForestClassifier(oob_score=True, random_state=42 , n_estimators=n_estimators)
        model.fit(X, y)
        logger.info(f"OOB Score: {model.oob_score_}")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def save_outputs(model: RandomForestClassifier, X_test_scaled: np.ndarray, metrics_path: str, model_path: str, test_data_path: str) -> None:
    logger.info("Saving model and evaluation data...")
    try:
        os.makedirs(metrics_path, exist_ok=True)

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save test data
        pd.DataFrame(X_test_scaled).to_csv(test_data_path, index=False)

        # Save metrics
        with open(os.path.join(metrics_path, "metrics"), "w") as f:
            json.dump({"oob_score": model.oob_score_}, f, indent=4)

        logger.info(f"Model and metrics saved in {metrics_path}")
    except Exception as e:
        logger.error(f"Error saving outputs: {e}")
        raise


def main():
    train_path = "./data/featured_data/train_data"
    test_path = "./data/featured_data/test_data"
    model_path = "./reports/model.pkl"
    metrics_path = os.path.join("./models", "evaluation_data")
    test_data_path = os.path.join("./data/interim", "X_test_scaled")

    try:
        train, test = load_data(train_path, test_path)
        y = train.iloc[:, 0]
        X = train.iloc[:, 1:]
        X_train_scaled, X_test_scaled, _ = scale_data(X, test)
        model = train_model(X_train_scaled, y)
        save_outputs(model, X_test_scaled, metrics_path, model_path, test_data_path)
    except Exception as e:
        logger.error(f"Model building pipeline failed: {e}")


if __name__ == "__main__":
    main()
