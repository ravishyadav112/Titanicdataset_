import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import OneHotEncoder

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading processed data...")
    try:
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        return train, test
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying feature engineering...")
    try:
        df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "male" else 1)
        df["Age"] = df["Age"].fillna(df["Age"].mean().round())
        df["Embarked"] = df["Embarked"].fillna("C")

        encoder = OneHotEncoder(drop="first")
        encoded = encoder.fit_transform(df[["Embarked"]]).toarray()
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["Embarked"]))

        df_final = pd.concat([df.drop("Embarked", axis=1).reset_index(drop=True), encoded_df], axis=1)
        return df_final
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


def save_data(train: pd.DataFrame, test: pd.DataFrame, output_dir: str) -> None:
    logger.info("Saving featured data...")
    try:
        os.makedirs(output_dir, exist_ok=True)
        train.to_csv(os.path.join(output_dir, "train_data"), index=False)
        test.to_csv(os.path.join(output_dir, "test_data"), index=False)
        logger.info(f"Data saved to {output_dir}")
    except Exception as e:
        logger.error(f"Error saving featured data: {e}")
        raise


def main():
    train_path = "./data/processed/train_processed_data"
    test_path = "./data/processed/test_processed_data"
    output_dir = os.path.join("data", "featured_data")

    try:
        train_raw, test_raw = load_data(train_path, test_path)
        train_feat = feature_engineering(train_raw)
        test_feat = feature_engineering(test_raw)
    
        save_data(train_feat, test_feat, output_dir)
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")


if __name__ == "__main__":
    main()
