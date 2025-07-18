import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_params(params_path:str)-> dict:
    """Load params from the yaml file"""
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters retrived from %s",params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found : %s",params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Yaml error :%s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error : %s",e)


def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int,
                text_col='text', target_col='target') -> tuple:
    """Apply TF-IDF to the text data and return train/test DataFrames with labels."""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data[text_col].values
        y_train = train_data[target_col].values
        X_test = test_data[text_col].values
        y_test = test_data[target_col].values

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        feature_names = vectorizer.get_feature_names_out()

        train_df = pd.DataFrame(X_train_tfidf.toarray(), columns=feature_names)
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_tfidf.toarray(), columns=feature_names)
        test_df['label'] = y_test

        logger.debug('TF-IDF applied. Train shape: %s, Test shape: %s', train_df.shape, test_df.shape)
        return train_df, test_df
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise


def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the DataFrame to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug('Data saved to %s', file_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        # max_features = 50
        params=load_params(params_path='params.yaml')
        max_features=params['feature_engineering']['max_features']
        train_data_path = './data/interim/train_processed.csv'
        test_data_path = './data/interim/test_processed.csv'

        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df, os.path.join('./data/processed', 'train_tfidf.csv'))
        save_data(test_df, os.path.join('./data/processed', 'test_tfidf.csv'))

        logger.debug('Feature engineering process completed successfully.')
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
