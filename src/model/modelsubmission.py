import pandas as pd
import pickle
import os
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str):
    logger.info("Loading trained model...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def make_predictions(model, X_test: pd.DataFrame) -> list:
    logger.info("Making predictions...")
    try:
        return model.predict(X_test)
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def save_submission(passenger_ids: pd.Series, predictions, output_path: str):
    logger.info("Saving submission file...")
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission = pd.DataFrame({
            "PassengerId": passenger_ids,
            "Survived": predictions
        })
        submission.to_csv(output_path, index=False)
        logger.info(f"Submission saved at {output_path}")
    except Exception as e:
        logger.error(f"Error saving submission: {e}")
        raise


def main():
    model_path = "./models/model.pkl"
    test_data_path = "./data/interim/X_test_scaled"
    id_data_path = "./data/raw/submission.csv"
    output_path = "./models/submission/submission_file.csv"

    try:
        model = load_model(model_path)
        X_test = pd.read_csv(test_data_path)
        id_df = pd.read_csv(id_data_path)
        predictions = make_predictions(model, X_test)
        save_submission(id_df["PassengerId"], predictions, output_path)
        logger.info("Submission pipeline completed successfully.")
    except Exception as e:
        logger.error(f"Submission pipeline failed: {e}")


if __name__ == "__main__":
    main()
