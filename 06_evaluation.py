import os
import argparse
import psycopg2
from dotenv import load_dotenv
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Read database connection details from environment variables
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute similarity scores using Word2Vec and BERT.")
parser.add_argument("--models_path", type=str, default="./versions/", help="Path to the directory containing models.")
args = parser.parse_args()

MODELS_PATH = args.models_path
BATCH_SIZE = 100  # Database query batch size
TOP_N = 50  # Number of top similar words to retrieve

# Load BERT model
bert_model = SentenceTransformer("microsoft/codebert-base")

# Connect to the PostgreSQL database
try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS similarity_results (
            model_name TEXT NOT NULL,
            criteria TEXT NOT NULL,
            similar_word TEXT NOT NULL,
            w2v_similarity_score FLOAT NOT NULL,
            bert_similarity_score FLOAT NOT NULL,
            PRIMARY KEY (model_name, criteria, similar_word)
        );
    """)
    conn.commit()
    print("✅ Successfully connected to the database.", flush=True)
except Exception as e:
    print(f"❌ Error connecting to the database: {e}", flush=True)
    exit()

# Get all available Word2Vec models in the directory
w2v_models = {
    fname: Word2Vec.load(os.path.join(MODELS_PATH, fname), mmap="r")
    for fname in os.listdir(MODELS_PATH) if fname.endswith(".model")
}

if not w2v_models:
    print("❌ No Word2Vec models found in the specified directory.", flush=True)
    exit()

print(f"✅ Loaded {len(w2v_models)} Word2Vec models.", flush=True)

# Paginate through the quality attributes table
offset = 0
while True:
    cursor.execute("SELECT attribute FROM quality_attributes ORDER BY attribute LIMIT %s OFFSET %s;", (BATCH_SIZE, offset))
    attributes = cursor.fetchall()

    if not attributes:
        break  # Exit loop when no more attributes

    for (attribute,) in attributes:
        attribute_ngram = attribute.replace(" ", "_")  # Adjust for Word2Vec token format

        for model_name, model in w2v_models.items():
            if attribute_ngram in model.wv:
                # Get top N similar words using Word2Vec
                similar_words = model.wv.most_similar(attribute_ngram, topn=TOP_N)

                for word, w2v_score in similar_words:
                    word_clean = word.replace("_", " ")  # Restore spaces for better readability

                    # Compute BERT similarity score
                    bert_score = util.cos_sim(bert_model.encode(attribute), bert_model.encode(word_clean)).item()

                    # Insert or update the similarity table
                    cursor.execute("""
                        INSERT INTO similarity_results (model_name, criteria, similar_word, w2v_similarity_score, bert_similarity_score)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (model_name, criteria, similar_word)
                        DO UPDATE SET w2v_similarity_score = EXCLUDED.w2v_similarity_score,
                                      bert_similarity_score = EXCLUDED.bert_similarity_score;
                    """, (model_name, attribute, word_clean, w2v_score, bert_score))

                    print(f"🔹 Model: {model_name} | Criteria: {attribute} | Word: {word_clean} | W2V: {w2v_score:.4f} | BERT: {bert_score:.4f}")

    # Move to next batch
    offset += BATCH_SIZE
    conn.commit()  # Commit after each batch

# Close database connection
cursor.close()
conn.close()
print("✅ All similarity scores updated successfully.", flush=True)
