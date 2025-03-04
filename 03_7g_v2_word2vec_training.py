import logging
import os
from tqdm import tqdm
from datetime import datetime
from gensim.models import Word2Vec
from database import (
    initialize_staging,
    fetch_tokenized_batches,
    last_processed_token_7g,
    update_last_processed_id_7g,
    save_model_to_db
)

# Word2Vec Config
W2V_MODEL_PATH = "stackoverflow_7g_v2_word2vec.model"
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 10
WORKERS = 4

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# **Function to Generate N-grams up to 7-words**
def generate_phrases(sentences, phrase_length=7): # Maximum length of phrases
    """
    Generates phrases up to `phrase_length` words from tokenized text.
    :param sentences: List of tokenized sentences (each sentence is a list of words)
    :param phrase_length: Maximum length of n-grams to generate
    :return: List of sentences with added n-grams
    """
    phrase_sentences = []  # Store sentences with their n-grams

    for sentence in sentences:
        new_sentence = sentence.copy()  # Copy original words
        for n in range(2, phrase_length + 1):  # Generate from 2-grams to max length
            ngrams = [
                "_".join(sentence[i:i + n])  # Join words with underscores
                for i in range(len(sentence) - n + 1)
            ]
            new_sentence.extend(ngrams)  # Append n-grams to sentence
        phrase_sentences.append(new_sentence)  # Store updated sentence

    return phrase_sentences
# **Function to Train Word2Vec with Phrases**
def train_word2vec():
    model = None
    model_version = 0  # Track model versioning

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
            update_last_processed_id_7g(last_processed_id)

    # **Train in Batches**
    start_id = last_processed_token_7g()
    total_rows = 0
    progress_bar = tqdm(desc="Training Word2Vec", unit=" rows", dynamic_ncols=True)

    initial_alpha = getattr(model, "alpha", 0.025) # Default initial learning rate
    min_alpha = 0.0001     # Minimum learning rate

    # Compute decay based on number of training iterations
    alpha_step = (initial_alpha - min_alpha) / 5  # 5 epochs

    for sentences, last_processed_id, total_processed in fetch_tokenized_batches(start_id=start_id):
        if len(sentences) > 0:
            # Generate phrases
            sentences_with_phrases = generate_phrases(sentences)

            # Update vocabulary and train the model
            model.build_vocab(sentences_with_phrases, update=True)  # Update vocabulary

            # Manually reset alpha to prevent "Effective 'alpha' higher than previous training cycles" warning
            model.alpha = initial_alpha  # Restore previous alpha
            model.min_alpha = min_alpha  # Restore min alpha
            
            # Train with controlled learning rate decay
            for epoch in range(5):  # Number of training epochs
                alpha = max(min_alpha, initial_alpha - epoch * alpha_step)  # Reduce alpha each epoch
                model.train(sentences_with_phrases, total_examples=len(sentences_with_phrases), epochs=1, start_alpha=alpha, end_alpha=min_alpha)

            # Save Model & Update Progress in DB
            model.save(W2V_MODEL_PATH)
            print(f"Saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", flush=True)
            update_last_processed_id_7g(last_processed_id)

            # Update progress bar
            progress_bar.update(total_processed - total_rows)
            total_rows = total_processed
            new_version = last_processed_id // 1000000
            if new_version > model_version:
                model_version = new_version
                # Save model to DB
                save_model_to_db(model, model_version)
                NEW_W2V_MODEL_PATH = "./versions/" + W2V_MODEL_PATH.replace(".model", f"_MV{model_version}.model")
                model.save(NEW_W2V_MODEL_PATH)
    
    progress_bar.close()

    logging.info("🎉 Training complete! Final model saved.")

# **Run Training**
if __name__ == "__main__":
    initialize_staging()
    train_word2vec()