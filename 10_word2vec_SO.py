import os
import time
from gensim.models import Word2Vec

SENTENCE_FILE_PATH = "./sentences/processed_sentences.txt"


class SentenceIterator:
    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line.split()  # Already tokenized, lowercased, and cleaned


def train_word2vec_from_file(filepath):
    print("✨ Initializing Word2Vec training from file...", flush=True)
    start = time.time()

    sentences = SentenceIterator(filepath)

    model = Word2Vec(
        vector_size=200,
        window=5,
        min_count=5,
        workers=4,
    )

    print("🔎 Building vocabulary...", flush=True)
    model.build_vocab(sentences)

    print("🚂 Training model...", flush=True)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=3,
    )

    end = time.time()
    print(f"🎉 Training complete in {(end - start) / 60:.2f} minutes.", flush=True)
    return model


def save_model(model):
    print("💾 Saving Word2Vec model...", flush=True)
    model.wv.save_word2vec_format("MDL_SO.txt", binary=False)
    model.wv.save_word2vec_format("MDL_SO.bin", binary=True)
    print("✅ Model saved successfully!", flush=True)


def main():
    print("🚀 Starting Word2Vec training from disk-backed sentences...", flush=True)
    model = train_word2vec_from_file(SENTENCE_FILE_PATH)
    save_model(model)


if __name__ == "__main__":
    main()
