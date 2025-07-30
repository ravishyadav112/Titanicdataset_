import numpy as np
import pandas as pd
import os
import logging

# Setup logger (console only)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load training and test data"""
    logger.info("Loading raw data...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        logger.info("Data loaded successfully.")
        return train, test
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data: fill missing, add features, drop columns"""
    logger.info("Processing data...")
    try:
        df["Age"] = df["Age"].fillna(df["Age"].mean().round())
        df["Embarked"] = df["Embarked"].fillna("C")
        df["Family_Member"] = df["Parch"] + df["SibSp"]
        df.drop(["SibSp", "Parch"], axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error during data processing: {e}")
        raise


def save_processed_data(train: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> None:
    """Save processed train and test data"""
    logger.info("Saving processed data...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, "train_processed_data"), index=False)
        test.to_csv(os.path.join(output_dir, "test_processed_data"), index=False)
        logger.info(f"Processed data saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def main():
    train_path = "./data/raw/train.csv"
    test_path = "./data/raw/test.csv"
    output_path = os.path.join("data", "processed")

    try:
        train_raw, test_raw = load_data(train_path, test_path)
        train_processed = process_data(train_raw)
        test_processed = process_data(test_raw)
        save_processed_data(train_processed, test_processed, output_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
