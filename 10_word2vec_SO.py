import time
import logging
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from database import read_cleaned_posts

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Configure logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class Sentences:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.sentences = []

    def process_data(self):
        last_processed_id = 0
        logging.info(f"Starting data processing from ID {last_processed_id}...")
        progress_bar = tqdm(desc="Processing", unit=" rows", dynamic_ncols=True)
        total_rows = 0
        
        while True:
            batch = read_cleaned_posts(batch_size=10000, start_id=last_processed_id)
            if not batch:
                break

            for row in batch:
                title = str(row[2]) if row[2] else ""
                body = str(row[3]) if row[3] else ""

                if title or body:  # Ensure at least one is not empty
                    tokens = word_tokenize(f"{title} {body}".lower())
                    tokens = [word for word in tokens if word.isalnum() and word not in self.stop_words]
                    self.sentences.append(" ".join(tokens))
                    total_rows += 1
                    last_processed_id = row[0]
            
            progress_bar.update(len(batch))  # Ensure progress bar updates correctly

        progress_bar.close()
        logging.info(f"Processed {total_rows} rows.")
        return self.sentences

if __name__ == "__main__":
    logging.info("🚀 Starting stopword removal and tokenization...")
    sentence_processor = Sentences()
    sentences = sentence_processor.process_data()

    # Save all sentences in a text file
    with open("processed_sentences.txt", "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")

    logging.info("✨ Training Word2Vec model...")
    w2v_start = time.time()
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=3)
    w2v_end = time.time()
    logging.info(f"🎉 Word2Vec training completed in {w2v_end - w2v_start:.2f} seconds.")

    logging.info("💾 Saving Word2Vec model...")
    model.wv.save_word2vec_format('MDL_SO.txt', binary=False)
    model.wv.save_word2vec_format("MDL_SO.bin", binary=True)
    logging.info("✅ Model saved successfully!")
