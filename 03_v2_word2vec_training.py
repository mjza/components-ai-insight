import logging
import os
from tqdm import tqdm
from datetime import datetime
from gensim.models import Word2Vec
from database import (
    initialize_staging,
    fetch_tokenized_batches,
    last_processed_token,
    update_last_processed_id,
    save_model_to_db
)

# Word2Vec Config
W2V_MODEL_PATH = "stackoverflow_v2_word2vec.model"
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 10
WORKERS = 4
NUM_EPOCHS = 5  # Number of training epochs

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


# **Function to Train Word2Vec**
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
            model.build_vocab(sentences)  # Build vocabulary from initial batch
            logging.info(f"✅ Vocabulary initialized with {len(sentences)} sentences.")
            update_last_processed_id(last_processed_id)

    # **Train in Batches**
    start_id = last_processed_token()
    total_rows = 0
    progress_bar = tqdm(desc="Training Word2Vec", unit=" rows", dynamic_ncols=True)

    for sentences, last_processed_id, total_processed in fetch_tokenized_batches(start_id=start_id):
        if len(sentences) > 0:
            # ⚠️ Save the previous alpha before updating vocab
            prev_alpha = model.alpha  
            prev_min_alpha = model.min_alpha  

            # Update vocabulary and train the model
            model.build_vocab(sentences, update=True)  # Update vocabulary

            # Manually reset alpha to prevent "Effective 'alpha' higher than previous training cycles" warning
            model.alpha = prev_alpha  # Restore previous alpha
            model.min_alpha = prev_min_alpha  # Restore min alpha
            
            # 🏁 Dynamically compute alpha decay
            
            alpha_step = (model.alpha - model.min_alpha) / NUM_EPOCHS

            # Train with controlled learning rate decay
            for epoch in range(NUM_EPOCHS):
                current_alpha = max(model.min_alpha, model.alpha - epoch * alpha_step)
                model.train(sentences, total_examples=len(sentences), epochs=1, start_alpha=current_alpha, end_alpha=model.min_alpha)
                

            # Save Model & Update Progress in DB
            model.save(W2V_MODEL_PATH)
            print(f"Saved at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.", flush=True)
            
            update_last_processed_id(last_processed_id)

            # Update progress bar
            progress_bar.update(total_processed - total_rows)
            total_rows = total_processed
            new_version = last_processed_id // 1000000
            if new_version > model_version:
                model_version = new_version
                # Save model to DB
                #save_model_to_db(model, model_version)
                NEW_W2V_MODEL_PATH = "./versions/" + W2V_MODEL_PATH.replace(".model", f"_MV{model_version}.model")
                model.save(NEW_W2V_MODEL_PATH)

    # Save model to DB
    #save_model_to_db(model, model_version)
    
    progress_bar.close()

    logging.info("🎉 Training complete! Final model saved.")


# **Run Training**
if __name__ == "__main__":
    initialize_staging()
    train_word2vec()
