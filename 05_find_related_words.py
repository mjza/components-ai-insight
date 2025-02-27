import os
from dotenv import load_dotenv
import psycopg2
from gensim.models import Word2Vec

# Load environment variables from .env file
load_dotenv()

# Read database connection details from environment variables
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DBC_NAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# Load the custom-trained Word2Vec model
model = Word2Vec.load("stackoverflow_7g_word2vec.model")

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
    print("✅ Successfully connected to the database.", flush=True)
except Exception as e:
    print(f"❌ Error connecting to the database: {e}", flush=True)
    exit()

# Define pagination variables
BATCH_SIZE = 100  # Number of rows per batch
offset = 0  # Start from the beginning
num = 1  # Counter for tracking processed attributes

# Paginate through the quality attributes table
while True:
    cursor.execute("SELECT attribute FROM quality_attributes ORDER BY attribute LIMIT %s OFFSET %s;", (BATCH_SIZE, offset))
    attributes = cursor.fetchall()

    if not attributes:
        break  # Exit loop when there are no more attributes to fetch

    # Process each attribute and find related words
    for (attribute,) in attributes:
        attribute_ngram = attribute.replace(" ", "_")  # Replace spaces with underscores

        if attribute_ngram in model.wv:
            # Find similar words with similarity > 0.5
            similar_words = model.wv.most_similar(attribute_ngram, topn=50)
            filtered_words = [word for word, similarity in similar_words if similarity > 0.5]

            # Update the related_words column in the database
            cursor.execute(
                "UPDATE quality_attributes SET related_words = %s WHERE attribute = %s;",
                (filtered_words, attribute)
            )
            print(f"🔹 {num}. Quality Criterion: {attribute} → {attribute_ngram}", flush=True)
            print(f"   Related Words: {', '.join(filtered_words)}", flush=True)
            num += 1

    # Move to the next batch
    offset += BATCH_SIZE
    conn.commit()  # Commit after each batch to avoid data loss

# Close the connection
cursor.close()
conn.close()

print("✅ All quality attributes updated with related words.", flush=True)
