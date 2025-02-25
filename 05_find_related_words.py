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
model = Word2Vec.load("stackoverflow_word2vec.model")

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

# Fetch all quality attributes from the table
cursor.execute("SELECT attribute FROM quality_attributes;")
attributes = cursor.fetchall()

# Process each attribute and find related words
for (attribute,) in attributes:
    if attribute in model.wv:
        # Find similar words with similarity > 0.9
        similar_words = model.wv.most_similar(attribute, topn=50)
        filtered_words = [word for word, similarity in similar_words if similarity > 0.9]

        # Update the related_words column in the database
        cursor.execute(
            "UPDATE quality_attributes SET related_words = %s WHERE attribute = %s;",
            (filtered_words, attribute)
        )
        print(f"🔹 Quality Criterion: {attribute}", flush=True)
        print(f"   Related Words: {', '.join(filtered_words)}", flush=True)

# Commit the changes and close the connection
conn.commit()
cursor.close()
conn.close()

print("✅ Quality attributes updated with related words.", flush=True)
