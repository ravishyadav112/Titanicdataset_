import pandas as pd
import numpy as np
import os
import logging

# Setup basic logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str, id_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and submission datasets"""
    logger.info("Loading data...")

    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        submission = pd.read_csv(id_path)
        logger.info("Data loaded successfully.")
        return train, test, submission
    except FileNotFoundError as e:
        logger.error(f"File not found: {e.filename}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary columns from the DataFrame"""
    logger.info("Cleaning data...")
    try:
        return df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    except KeyError as e:
        logger.error(f"Missing columns during drop: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while cleaning data: {e}")
        raise


def save_data(train: pd.DataFrame, test: pd.DataFrame, submission: pd.DataFrame, output_dir: str) -> None:
    """Save the cleaned data to the specified directory"""
    logger.info("Saving cleaned data...")

    try:
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
        submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
        logger.info(f"Data saved to {output_dir}")
    except PermissionError as e:
        logger.error(f"Permission denied while saving files: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while saving data: {e}")
        raise


def main():
    # Define file paths
    train_path = r"C:\Users\ASUS\Titanic\train.csv"
    test_path = r"C:\Users\ASUS\Titanic\test.csv"
    id_path = r"C:\Users\ASUS\Titanic\gender_submission.csv"
    output_path = os.path.join("data", "raw")

    try:
        train, test, submission = load_data(train_path, test_path, id_path)
        train_clean = clean_data(train)
        test_clean = clean_data(test)
        save_data(train_clean, test_clean, submission, output_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    main()
