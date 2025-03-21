import os
import sys
import time
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from database import initialize_staging, read_cleaned_posts, update_last_processed_id, last_processed_token

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

SENTENCE_FILE_PATH = "./sentences/processed_sentences.txt"

class Sentences:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.sentences = []

        # Ensure sentences directory exists
        os.makedirs("./sentences", exist_ok=True)

        # Load existing sentences if file exists
        if os.path.exists(SENTENCE_FILE_PATH):
            print(f"📄 Loadeding sentences from existing file.", flush=True)
            with open(SENTENCE_FILE_PATH, "r", encoding="utf-8") as f:
                self.sentences = [line.strip() for line in f if line.strip()]
            print(f"📄 Loaded {len(self.sentences)} existing sentences from file.", flush=True)

    def process_data(self):
        last_processed_id = last_processed_token()
        print(f"🚧 Starting data processing from ID {last_processed_id}...", flush=True)
        progress_bar = tqdm(desc="Processing", unit=" rows", dynamic_ncols=True)
        total_rows = 0

        while True:
            batch = read_cleaned_posts(batch_size=10000, start_id=last_processed_id)
            if not batch:
                break

            new_sentences = []

            for row in batch:
                title = str(row[2]) if row[2] else ""
                body = str(row[3]) if row[3] else ""

                if title or body:
                    tokens = word_tokenize(f"{title} {body}".lower())
                    tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
                    sentence = " ".join(tokens)
                    self.sentences.append(sentence)
                    new_sentences.append(sentence)
                    total_rows += 1
                    last_processed_id = row[0]

            if new_sentences:
                # Append new sentences to file
                with open(SENTENCE_FILE_PATH, "a", encoding="utf-8") as f:
                    for sentence in new_sentences:
                        f.write(sentence + "\n")
                    f.flush()

                update_last_processed_id(last_processed_id)

            progress_bar.update(len(batch))

        progress_bar.close()
        print(f"✅ Processed {total_rows} new rows.", flush=True)
        return self.sentences

if __name__ == "__main__":
    print("🚀 Starting stopword removal and tokenization...", flush=True)
    initialize_staging()

    sentence_processor = Sentences()
    sentences = sentence_processor.process_data()

    print("✨ Training Word2Vec model...", flush=True)
    w2v_start = time.time()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=3)
    w2v_end = time.time()
    print(f"🎉 Word2Vec training completed in {w2v_end - w2v_start:.2f} seconds.", flush=True)

    print("💾 Saving Word2Vec model...", flush=True)
    model.wv.save_word2vec_format("MDL_SO.txt", binary=False)
    model.wv.save_word2vec_format("MDL_SO.bin", binary=True)
    print("✅ Model saved successfully!", flush=True)
