import logging
import os
from tqdm import tqdm
from gensim.models import Word2Vec
from database import (
    initialize_staging,
    fetch_tokenized_batches,
    last_processed_token,
    update_last_processed_id,
    save_model_to_db
)

# Word2Vec Config
W2V_MODEL_PATH = "stackoverflow_word2vec.model"
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 5
WORKERS = 4

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


# **Function to Train Word2Vec**
def train_word2vec():
    model = None
    model_version = 1  # Track model versioning

    # Load Existing Model if Available
    if os.path.exists(W2V_MODEL_PATH):
        logging.info("🔄 Loading existing Word2Vec model...")
        model = Word2Vec.load(W2V_MODEL_PATH)

        # Get latest model version
        model_version = last_processed_token() + 1
    else:
        logging.info("🆕 Creating new Word2Vec model...")
        model = Word2Vec(vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS)

        # Build the vocabulary with the first batch
        initial_batch = next(fetch_tokenized_batches(batch_size=10000, start_id=0), None)
        if initial_batch:
            sentences, last_processed_id, total_processed = initial_batch
            model.build_vocab(sentences)  # Build vocabulary from initial batch
            logging.info(f"✅ Vocabulary initialized with {len(sentences)} sentences.")
            update_last_processed_id(last_processed_id)

    # **Train in Batches**
    start_id = last_processed_token()
    total_rows = 0
    progress_bar = tqdm(desc="Training Word2Vec", unit=" rows", dynamic_ncols=True)

    for sentences, last_processed_id, total_processed in fetch_tokenized_batches(start_id=start_id):
        if len(sentences) > 0:
            model.build_vocab(sentences, update=True)  # Update the vocabulary
            model.train(sentences, total_examples=len(sentences), epochs=5)  # Train the model

            # Save Model & Update Progress in DB
            model.save(W2V_MODEL_PATH)
            
            update_last_processed_id(last_processed_id)

            # Update progress bar
            progress_bar.update(total_processed - total_rows)
            total_rows = total_processed

    # Save model to DB
    save_model_to_db(model, model_version)
    
    progress_bar.close()

    logging.info("🎉 Training complete! Final model saved.")


# **Run Training**
if __name__ == "__main__":
    initialize_staging()
    train_word2vec()
