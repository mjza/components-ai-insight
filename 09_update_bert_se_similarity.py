import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Load environment variables
load_dotenv()

# Read database connection details from environment variables
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

BATCH_SIZE = 1000  # Adjust batch size for efficiency

# Load BERT_SE model
BERT_SE_PATH = "./BERT_SE"

try:
    print(f"📥 Loading BERT_SE model from {BERT_SE_PATH}...", flush=True)
    bert_se_model = SentenceTransformer(BERT_SE_PATH)
    print("✅ Successfully loaded BERT_SE model.", flush=True)
except Exception as e:
    print(f"❌ Error loading BERT_SE model: {e}", flush=True)
    exit()

# Connect to PostgreSQL database
try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    
    print("✅ Successfully connected to the database.", flush=True)

except Exception as e:
    print(f"❌ Error connecting to the database: {e}", flush=True)
    exit()

# Process records where bert_se_similarity_score is NULL
offset = 0

while True:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_name, criteria, similar_word 
        FROM similarity_results 
        WHERE bert_se_similarity_score IS NULL
        ORDER BY model_name, criteria, similar_word
        LIMIT %s OFFSET %s;
    """, (BATCH_SIZE, offset))
    
    rows = cursor.fetchall()

    if not rows:
        break  # Exit loop if no more rows to process

    print(f"📥 Processing {len(rows)} rows...", flush=True)

    for model_name, criteria, similar_word in rows:
        try:
            # Compute similarity using SentenceTransformer
            bert_se_score = util.cos_sim(
                bert_se_model.encode(criteria, convert_to_tensor=True),
                bert_se_model.encode(similar_word, convert_to_tensor=True)
            ).item()

            cursor.execute("""
                UPDATE similarity_results 
                SET bert_se_similarity_score = %s 
                WHERE criteria = %s AND similar_word = %s;
            """, (bert_se_score, criteria, similar_word))
        
        except Exception as e:
            print(f"❌ Error processing ({criteria}, {similar_word}): {e}", flush=True)
            continue

    # Move to next batch
    offset += BATCH_SIZE

    # Commit after processing a batch
    conn.commit()

print("✅ bert_se_similarity_score updated successfully.", flush=True)

# Close database connection
cursor.close()
conn.close()
