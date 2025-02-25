import logging
import os
from tqdm import tqdm
from gensim.models import Word2Vec, Phrases
from gensim.models.phrases import Phraser
from database import (
    fetch_tokenized_batches
)

# Word2Vec Config
W2V_MODEL_PATH = "stackoverflow_7g_word2vec.model"
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 5
WORKERS = 4
PHRASE_LENGTH = 7  # Maximum length of phrases

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


# **Function to Generate N-grams up to 7-words**
def generate_phrases(sentences):
    """
    Generates phrases up to PHRASE_LENGTH words from tokenized text.
    """
    phrases = sentences.copy()

    for n in range(2, PHRASE_LENGTH + 1):  # Generate from 2-grams to 7-grams
        ngram_phrases = []
        for sentence in sentences:
            ngrams = [
                "_".join(sentence[i:i + n])  # Join words with underscores
                for i in range(len(sentence) - n + 1)
            ]
            ngram_phrases.append(sentence + ngrams)  # Combine words with n-grams
        phrases = ngram_phrases

    return phrases


# **Function to Train Word2Vec with Phrases**
def train_word2vec():
    model = None

    # Load Existing Model if Available
    if os.path.exists(W2V_MODEL_PATH):
        logging.info("🔄 Loading existing Word2Vec model...")
        model = Word2Vec.load(W2V_MODEL_PATH)

    else:
        logging.info("🆕 Creating new Word2Vec model...")
        model = Word2Vec(vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS)

        # Build the vocabulary with the first batch
        initial_batch = next(fetch_tokenized_batches(batch_size=10000, start_id=0), None)
        if initial_batch:
            sentences, last_processed_id, total_processed = initial_batch
            # Generate phrases from the initial batch
            sentences_with_phrases = generate_phrases(sentences)
            model.build_vocab(sentences_with_phrases)  # Build vocabulary from initial batch
            logging.info(f"✅ Vocabulary initialized with {len(sentences_with_phrases)} sentences.")

    # **Train in Batches**
    start_id = 0
    total_rows = 0
    progress_bar = tqdm(desc="Training Word2Vec", unit=" rows", dynamic_ncols=True)

    for sentences, last_processed_id, total_processed in fetch_tokenized_batches(start_id=start_id):
        if len(sentences) > 0:
            # Generate phrases
            sentences_with_phrases = generate_phrases(sentences)

            # Update vocabulary and train the model
            model.build_vocab(sentences_with_phrases, update=True)  # Update vocabulary
            model.train(sentences_with_phrases, total_examples=len(sentences_with_phrases), epochs=5)

            # Save Model & Update Progress in DB
            model.save(W2V_MODEL_PATH)

            # Update progress bar
            progress_bar.update(total_processed - total_rows)
            total_rows = total_processed

    progress_bar.close()

    logging.info("🎉 Training complete! Final model saved.")


# **Run Training**
if __name__ == "__main__":
    train_word2vec()
