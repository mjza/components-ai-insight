import logging
from tqdm import tqdm
import sys
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from database import (
    initialize_staging,
    insert_into_tokenized_posts,
    last_tokenized_post,
    read_cleaned_posts
)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


# **Function to Preprocess Text (Title + Body)**
def preprocess_text(title, body):
    """
    Tokenizes and cleans text by:
    - Lowercasing
    - Removing non-alphanumeric words
    - Removing stopwords
    - Returning space-separated tokens
    - Combines title + body for better representation
    """
    stop_words = set(stopwords.words("english"))
    combined_text = f"{title} {body}"  # Concatenating title and body

    tokens = word_tokenize(combined_text.lower())  # Tokenize after converting to lowercase
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stopwords

    return " ".join(tokens), tokens  # Store as space-separated text and Return both formats  

   
def process_and_store_tokens():
    last_processed_id = last_tokenized_post()  # Fetch last processed ID from DB
    total_rows = 0  # Keep track of processed rows

    logging.info(f"🚀 Starting tokenization from ID {last_processed_id}...\n")

    # Use tqdm for progress bar
    progress_bar = tqdm(desc="Tokenizing", unit=" rows", dynamic_ncols=True)

    for batch in read_cleaned_posts(batch_size=10000, start_id=last_processed_id):
        processed_data = [
            (row[0], *preprocess_text(row[2], row[3]))  # row[2] = title, row[3] = body
            for row in batch if (row[2] or row[3]) # Ensure title or body exists
        ]

        if not processed_data:
            logging.info("\n✅ No more rows to process. Tokenization completed.")
            break

        insert_into_tokenized_posts(processed_data)  # Insert batch into DB
        last_processed_id = processed_data[-1][0]  # Update last processed ID

        total_rows += len(processed_data)  # Update total count
        progress_bar.update(len(processed_data))  # Update tqdm progress bar

    progress_bar.close()  # Close progress bar after completion
    logging.info("🎉 Tokenization process completed!")


# **Run Tokenization**
if __name__ == "__main__":
    initialize_staging()  # Ensure DB is ready
    process_and_store_tokens()  # Start tokenization
