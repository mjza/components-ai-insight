import os
import psycopg2
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv
from transformers import BertTokenizer, TFBertModel

# Load environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DBC_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

BATCH_SIZE = 100
BERT_SE_PATH = "./BERT_SE"

# Load tokenizer and model from checkpoint
bert_tokenizer = BertTokenizer.from_pretrained(BERT_SE_PATH)
bert_model = TFBertModel.from_pretrained(BERT_SE_PATH)

# Function to compute similarity
def compute_similarity(text1, text2):
    inputs_1 = bert_tokenizer(text1, return_tensors="tf", padding=True, truncation=True, max_length=512)
    inputs_2 = bert_tokenizer(text2, return_tensors="tf", padding=True, truncation=True, max_length=512)

    # Get embeddings
    with tf.device('/CPU:0'):  # Use CPU if GPU is not available
        embeddings_1 = bert_model(**inputs_1).last_hidden_state[:, 0, :]
        embeddings_2 = bert_model(**inputs_2).last_hidden_state[:, 0, :]

    # Compute cosine similarity
    similarity = np.dot(embeddings_1, embeddings_2.T) / (np.linalg.norm(embeddings_1) * np.linalg.norm(embeddings_2))
    return float(similarity.numpy())

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
    
    print("✅ Connected to the database.", flush=True)

    # Fetch rows where bert_se_similarity_score is NULL
    cursor.execute("""
        SELECT model_name, criteria, similar_word 
        FROM similarity_results 
        WHERE bert_se_similarity_score IS NULL
        LIMIT %s;
    """, (BATCH_SIZE,))
    
    rows = cursor.fetchall()
    
    if not rows:
        print("✅ No rows to update. Exiting...", flush=True)
        cursor.close()
        conn.close()
        exit()

    print(f"📥 Processing {len(rows)} rows...", flush=True)

    # Process each row and update similarity score
    for model_name, criteria, similar_word in rows:
        try:
            score = compute_similarity(criteria, similar_word)

            cursor.execute("""
                UPDATE similarity_results 
                SET bert_se_similarity_score = %s 
                WHERE model_name = %s AND criteria = %s AND similar_word = %s;
            """, (score, model_name, criteria, similar_word))
        
        except Exception as e:
            print(f"❌ Error processing ({criteria}, {similar_word}): {e}", flush=True)
            continue

    conn.commit()
    print("✅ bert_se_similarity_score updated successfully.", flush=True)

except Exception as e:
    print(f"❌ Database connection error: {e}", flush=True)

finally:
    cursor.close()
    conn.close()
    print("✅ Connection closed.", flush=True)
