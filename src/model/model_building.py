import pandas as pd
import numpy as np
import os
import json
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml



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


def train_model(X: np.ndarray, y: pd.Series) ->  XGBClassifier:
    logger.info("Training model...")
    try:
        model = XGBClassifier()
        model.fit(X, y)
        logger.info("Model Trained ...")
        return model
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise


def save_outputs(model: XGBClassifier, X_test_scaled: np.ndarray,  model_path: str, test_data_path: str) -> None:
    logger.info("Saving model and evaluation data...")
    try:

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Save test data
        pd.DataFrame(X_test_scaled).to_csv(os.path.join(test_data_path , "X_test_scaled"), index=False)

    except Exception as e:
        logging.error(f"Error in loading model{e}")



def main():
    train_path = "./data/featured_data/train_data"
    test_path = "./data/featured_data/test_data"
    model_path = "./reports/model.pkl"
    test_data_path ="./data/interim"

    try:
        train, test = load_data(train_path, test_path)
        y = train.iloc[:, 0]
        X = train.iloc[:, 1:]
        X_train_scaled, X_test_scaled, _ = scale_data(X, test)
        model = train_model(X_train_scaled, y)
        save_outputs(model, X_test_scaled, model_path, test_data_path)
    except Exception as e:
        logger.error(f"Model building pipeline failed: {e}")


if __name__ == "__main__":
    main()
