import os
import time
from tqdm import tqdm
from gensim.models import Word2Vec
from database import initialize_staging, fetch_tokenized_batches, update_last_processed_id, last_processed_token


SENTENCE_FILE_PATH = "./sentences/processed_sentences.txt"


def ensure_sentence_dir():
    os.makedirs("./sentences", exist_ok=True)


def export_sentences_to_file():
    ensure_sentence_dir()
    last_id = last_processed_token()
    print(f"🚧 Exporting data starting from ID {last_id}...", flush=True)

    progress_bar = tqdm(desc="Exporting", unit=" rows", dynamic_ncols=True)
    total_rows = 0

    while True:
        batch = fetch_tokenized_batches(batch_size=10000, start_id=last_id)
        if not batch:
            break

        new_sentences = []

        for row in batch:
            text = str(row[1]) if row[1] else ""
            if text:
                new_sentences.append(text)
                last_id = row[0]
                total_rows += 1

        if new_sentences:
            with open(SENTENCE_FILE_PATH, "a", encoding="utf-8") as f:
                for sentence in new_sentences:
                    f.write(sentence + "\n")
                f.flush()

            update_last_processed_id(last_id)

        progress_bar.update(len(batch))

    progress_bar.close()
    print(f"✅ Exported {total_rows} new rows to file.", flush=True)


def load_sentences_from_file():
    if not os.path.exists(SENTENCE_FILE_PATH):
        raise FileNotFoundError(f"Sentence file not found at {SENTENCE_FILE_PATH}")
    
    print("📄 Loading sentences from file...", flush=True)
    with open(SENTENCE_FILE_PATH, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"📄 Loaded {len(sentences)} sentences.", flush=True)
    return sentences


def train_word2vec(sentences):
    print("✨ Training Word2Vec model...", flush=True)
    start = time.time()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=3)
    duration = time.time() - start
    print(f"🎉 Word2Vec training completed in {duration:.2f} seconds.", flush=True)
    return model


def save_model(model):
    print("💾 Saving Word2Vec model...", flush=True)
    model.wv.save_word2vec_format("MDL_SO.txt", binary=False)
    model.wv.save_word2vec_format("MDL_SO.bin", binary=True)
    print("✅ Model saved successfully!", flush=True)


def main():
    print("🚀 Starting sentence export and model training pipeline...", flush=True)
    initialize_staging()

    # Step 1: Export from DB to file (no memory storage)
    export_sentences_to_file()

    # Step 2: Load sentences from file into memory
    sentences = load_sentences_from_file()

    # Step 3: Train and save Word2Vec model
    model = train_word2vec(sentences)
    save_model(model)


if __name__ == "__main__":
    main()
