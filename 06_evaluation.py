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
DB_NAME = os.getenv("DBC_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Load MODELS_PATH from environment variable
MODELS_PATH = os.getenv("MODELS_PATH", "").strip()

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compute similarity scores using Word2Vec and BERT.")
parser.add_argument("--models_path", type=str, default="./versions/", help="Path to the directory containing models.")
args = parser.parse_args()

# If MODELS_PATH is empty, use the command-line argument
if not MODELS_PATH:
    MODELS_PATH = args.models_path

BATCH_SIZE = 1000  # Database query batch size
TOP_N = 200  # Number of top similar words to retrieve

# Load BERT model once
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
    
    # Ensure the table exists
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

    # Get already processed model names
    cursor.execute("SELECT DISTINCT model_name FROM similarity_results;")
    processed_models = {row[0] for row in cursor.fetchall()}

except Exception as e:
    print(f"❌ Error connecting to the database: {e}", flush=True)
    exit()

# Get all available Word2Vec models in the directory
model_files = [fname for fname in os.listdir(MODELS_PATH) if fname.endswith(".model")]

# Remove already processed models
model_files = [fname for fname in model_files if fname.replace(".model", "") not in processed_models]

if not model_files:
    print("✅ All models are already processed. Exiting...", flush=True)
    cursor.close()
    conn.close()
    exit()

print(f"✅ Found {len(model_files)} new Word2Vec models. Processing one at a time.", flush=True)

# Process each model separately
for model_file in model_files:
    model_path = os.path.join(MODELS_PATH, model_file)
    model_name = model_file.replace(".model", "")

    print(f"📥 Loading model: {model_name}", flush=True)
    
    try:
        model = Word2Vec.load(model_path, mmap="r")
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}", flush=True)
        continue  # Skip model if it fails to load

    print(f"✅ Successfully loaded {model_name}. Processing criteria...", flush=True)

    # Paginate through the quality attributes table
    offset = 0
    while True:
        cursor = conn.cursor()
        cursor.execute("SELECT attribute FROM quality_attributes ORDER BY attribute LIMIT %s OFFSET %s;", (BATCH_SIZE, offset))
        attributes = cursor.fetchall()

        if not attributes:
            break  # Exit loop when no more attributes

        for (attribute,) in attributes:
            attribute_ngram = attribute.replace(" ", "_")  # Adjust for Word2Vec token format

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
                        ON CONFLICT (model_name, criteria, similar_word) DO NOTHING;
                    """, (model_name, attribute, word_clean, w2v_score, bert_score))

        # Move to next batch
        offset += BATCH_SIZE

    # commit at the end of the processing of a model.      
    conn.commit()

    print(f"✅ Finished processing {model_name}. Moving to the next model...\n", flush=True)

# Close database connection
cursor.close()
conn.close()
print("✅ All similarity scores updated successfully.", flush=True)
