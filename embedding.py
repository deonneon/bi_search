"""
Doc2Vec Embedding for MicroStrategy Metadata

This script uses Gensim's Doc2Vec to create document embeddings for preprocessed
MicroStrategy metadata artifacts.
"""

import pandas as pd
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from data_preprocessing import preprocess_data
from microstrategy_ingestion import MicroStrategyDummyIngestion

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_corpus(df: pd.DataFrame) -> list[TaggedDocument]:
    """
    Prepare the corpus for Doc2Vec training from the preprocessed DataFrame.

    Args:
        df: The preprocessed DataFrame with cleaned text columns.

    Returns:
        A list of TaggedDocument objects.
    """
    logger.info("Preparing corpus for Doc2Vec...")
    
    # The 'combined_text' is a string, let's use the tokenized columns
    df['tokens'] = df['cleaned_name'] + df['cleaned_description']
    
    corpus = [
        TaggedDocument(words=row['tokens'], tags=[str(i)])
        for i, row in df.iterrows()
        if row['tokens']
    ]
    
    if not corpus:
        logger.warning("Corpus is empty. No documents to train on.")
    
    logger.info(f"Corpus prepared with {len(corpus)} documents.")
    return corpus

def train_and_save_doc2vec(corpus: list[TaggedDocument], model_path: str = "doc2vec.model", vector_path: str = "doc_vectors.csv"):
    """
    Train a Doc2Vec model and save the model and its vectors.

    Args:
        corpus: The list of TaggedDocument objects for training.
        model_path: Path to save the trained Doc2Vec model.
        vector_path: Path to save the inferred document vectors.
    """
    if not corpus:
        logger.error("Cannot train model on an empty corpus.")
        return

    logger.info("Training Doc2Vec model...")
    # Initialize model - using parameters from the task description
    model = Doc2Vec(
        vector_size=300,
        min_count=2,
        epochs=40,
        workers=4 # Use 4 workers for faster training
    )

    # Build vocabulary
    model.build_vocab(corpus)

    # Train the model
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the trained model
    model.save(model_path)
    logger.info(f"Doc2Vec model saved to {model_path}")

    # Save the inferred vectors
    doc_vectors = {tag: model.dv[tag] for tag in model.dv.index_to_key}
    vector_df = pd.DataFrame.from_dict(doc_vectors, orient='index')
    vector_df.to_csv(vector_path)
    logger.info(f"Document vectors saved to {vector_path}")

if __name__ == '__main__':
    # 1. Ingest dummy data
    logger.info("Ingesting dummy data from MicroStrategy...")
    ingestion_client = MicroStrategyDummyIngestion()
    if ingestion_client.authenticate():
        metadata = ingestion_client.fetch_all_metadata()
        raw_df = ingestion_client.metadata_to_dataframe(metadata)

        if not raw_df.empty:
            # 2. Preprocess the data
            preprocessed_df = preprocess_data(raw_df)

            # 3. Prepare the corpus for Doc2Vec
            corpus = prepare_corpus(preprocessed_df)
            
            if corpus:
                print("\nSample of prepared corpus (first 3 documents):")
                for i in range(min(3, len(corpus))):
                    print(corpus[i])
                
                # 4. Train and save the model and vectors
                train_and_save_doc2vec(corpus) 