"""
Data Preprocessing for MicroStrategy Metadata

This script provides functions to clean, normalize, and preprocess the metadata
ingested from MicroStrategy to prepare it for analysis and topic modeling.
"""

import pandas as pd
import re
import logging
from typing import List, Optional
from gensim.parsing.preprocessing import STOPWORDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_text(text: Optional[str]) -> List[str]:
    """
    Normalize text by converting to lowercase, removing punctuation, and tokenizing.

    Args:
        text: The input string to normalize.

    Returns:
        A list of cleaned and tokenized words.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = text.split()
        return tokens
    except Exception as e:
        logger.error(f"Error normalizing text: {text} - {e}")
        return []

def remove_stopwords(tokens: List[str], custom_stopwords: Optional[List[str]] = None) -> List[str]:
    """
    Remove stopwords from a list of tokens.

    Args:
        tokens: A list of text tokens.
        custom_stopwords: An optional list of custom stopwords to add.

    Returns:
        A list of tokens with stopwords removed.
    """
    if not tokens:
        return []
        
    all_stopwords = STOPWORDS
    if custom_stopwords:
        all_stopwords = all_stopwords.union(set(custom_stopwords))
    
    return [word for word in tokens if word not in all_stopwords]

def flag_antipatterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag common anti-patterns in the metadata.
    
    - Empty descriptions
    - Generic names (e.g., 'Copy of...', 'test')
    - Versioning in names (e.g., 'v1', '_old')
    
    Args:
        df: The input DataFrame.
        
    Returns:
        The DataFrame with added boolean flag columns for anti-patterns.
    """
    logger.info("Flagging anti-patterns...")
    
    # Flag empty or whitespace-only descriptions
    df['is_empty_description'] = df['description'].str.strip().fillna('').eq('')

    # Flag generic names
    generic_patterns = r'\b(copy of|test|sample|draft|backup|old|v\d)\b'
    df['is_generic_name'] = df['name'].str.contains(generic_patterns, case=False, na=False)

    logger.info("Finished flagging anti-patterns.")
    return df
    
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full preprocessing pipeline to the metadata DataFrame.

    Args:
        df: The raw metadata DataFrame.

    Returns:
        The preprocessed DataFrame.
    """
    if df.empty:
        logger.warning("Input DataFrame is empty. Skipping preprocessing.")
        return df

    logger.info("Starting data preprocessing pipeline...")
    
    processed_df = df.copy()

    # Normalize name and description fields
    logger.info("Normalizing and cleaning text fields...")
    processed_df['cleaned_name'] = processed_df['name'].apply(normalize_text).apply(remove_stopwords)
    processed_df['cleaned_description'] = processed_df['description'].apply(normalize_text).apply(remove_stopwords)
    
    # Combine cleaned text fields for embedding
    processed_df['combined_text'] = processed_df['cleaned_name'] + processed_df['cleaned_description']
    processed_df['combined_text'] = processed_df['combined_text'].apply(' '.join)

    # Flag anti-patterns
    processed_df = flag_antipatterns(processed_df)
    
    logger.info("Data preprocessing pipeline finished.")
    return processed_df

if __name__ == '__main__':
    # Example usage with dummy data
    from microstrategy_ingestion import MicroStrategyDummyIngestion

    # 1. Ingest dummy data
    ingestion_client = MicroStrategyDummyIngestion()
    if ingestion_client.authenticate():
        metadata = ingestion_client.fetch_all_metadata()
        raw_df = ingestion_client.metadata_to_dataframe(metadata)

        if not raw_df.empty:
            # 2. Preprocess the data
            preprocessed_df = preprocess_data(raw_df)

            # 3. Display summary
            print("Original DataFrame sample:")
            print(raw_df[['name', 'description']].head())
            print("\nPreprocessed DataFrame sample:")
            print(preprocessed_df[['name', 'cleaned_name', 'description', 'cleaned_description', 'is_empty_description', 'is_generic_name']].head())
            
            print(f"\nTotal rows: {len(preprocessed_df)}")
            print(f"Empty descriptions found: {preprocessed_df['is_empty_description'].sum()}")
            print(f"Generic names found: {preprocessed_df['is_generic_name'].sum()}") 