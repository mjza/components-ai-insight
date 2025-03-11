import os
import argparse
import psycopg2
from dotenv import load_dotenv
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer, util

# Load environment variables from .env file
load_dotenv()

# Read database connection details from environment variables
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DBC_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

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

except Exception as e:
    print(f"❌ Error connecting to the database: {e}", flush=True)
    exit()

# Load the pre-trained Word2Vec model
w2v_model_path = "./models_ref/SO_vectors_200.bin"

try:
    print(f"📥 Loading Word2Vec model from {w2v_model_path}", flush=True)
    word2vec = KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)
    print("✅ Successfully loaded Word2Vec model.", flush=True)
except Exception as e:
    print(f"❌ Error loading Word2Vec model: {e}", flush=True)
    exit()

# Process quality attributes from the database
offset = 0
model_name = "SO_vectors_200"

while True:
    cursor = conn.cursor()
    cursor.execute("SELECT attribute FROM quality_attributes ORDER BY attribute LIMIT %s OFFSET %s;", (BATCH_SIZE, offset))
    attributes = cursor.fetchall()

    if not attributes:
        break  # Exit loop when no more attributes

    for (attribute,) in attributes:
        attribute_ngram = attribute.replace(" ", "_")  # Adjust for Word2Vec token format

        if attribute_ngram in word2vec:
            # Get top N similar words using Word2Vec
            similar_words = word2vec.most_similar(attribute_ngram, topn=TOP_N)

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

    # Commit after processing a batch
    conn.commit()

print("✅ All similarity scores updated successfully.", flush=True)

# Close database connection
cursor.close()
conn.close()
