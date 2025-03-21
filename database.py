import os
import json
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import logging
import pickle

# Configure logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Database credentials
DB_USER = os.getenv("DB_USER")
DB_HOST = os.getenv("DB_HOST")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

DBS_NAME = os.getenv("DBS_NAME")  # Stack Overflow (READ ONLY)
DBL_NAME = os.getenv("DBL_NAME")  # Libraries.io (READ ONLY)
DBC_NAME = os.getenv("DBC_NAME")  # Components Insight (WRITE - STAGING TABLES)


# ---------------------------- DATABASE CONNECTION FUNCTION ----------------------------

def get_connection(db_name):
    """Establishes a PostgreSQL database connection for the given database name."""
    try:
        return psycopg2.connect(
            dbname=db_name,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    except Exception as e:
        print(f"❌ Error connecting to database {db_name}: {e}")
        return None


# ---------------------------- PAGINATED READ FUNCTIONS ----------------------------

def read_stackoverflow_posts(batch_size=10000, start_id=0):
    """
    Reads posts from Stack Overflow database in paginated batches.
    :param batch_size: Number of records per batch (default: 10,000)
    :yield: Batch of records
    """
    conn = get_connection(DBS_NAME)
    if not conn:
        return

    cur = conn.cursor()
    offset = 0

    while True:
        cur.execute(
            f"SELECT id, posttypeid, title, body, tags FROM public.posts_md WHERE id > {start_id} ORDER BY id LIMIT %s OFFSET %s;",
            (batch_size, offset),
        )
        rows = cur.fetchall()

        if not rows:
            break  # No more data

        yield rows
        offset += batch_size  # Move to next batch

    cur.close()
    conn.close()


def read_libraries_projects(batch_size=10000):
    """
    Reads projects from Libraries.io database in paginated batches.
    :param batch_size: Number of records per batch (default: 10,000)
    :yield: Batch of records
    """
    conn = get_connection(DBL_NAME)
    if not conn:
        return

    cur = conn.cursor()
    offset = 0

    while True:
        cur.execute(
            """
            SELECT 
                id, name, platform, description 
            FROM 
                public.projects 
            WHERE (
                jsonb_typeof(raw) = 'object' 
                AND 
                (SELECT COUNT(*) FROM jsonb_object_keys(raw)) > 1
            )
            ORDER BY id LIMIT %s OFFSET %s;
            """,
            (batch_size, offset),
        )
        rows = cur.fetchall()

        if not rows:
            break  # No more data

        yield rows
        offset += batch_size  # Move to next batch

    cur.close()
    conn.close()


def read_cleaned_posts(batch_size=10000, start_id=0):
    """
    Reads projects from Libraries.io database in paginated batches.
    :param batch_size: Number of records per batch (default: 10,000)
    :yield: Batch of records
    """
    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    offset = 0

    while True:
        cur.execute(
            f"SELECT id, posttypeid, title, body, tags, extracted_libraries FROM public.stage_posts_cleaned WHERE id > {start_id} ORDER BY id LIMIT %s OFFSET %s;",
            (batch_size, offset),
        )
        rows = cur.fetchall()

        if not rows:
            break  # No more data

        yield rows
        offset += batch_size  # Move to next batch

    cur.close()
    conn.close()
    

# **Function to Fetch Tokenized Data in Batches**
def fetch_tokenized_batches(batch_size=10000, start_id=0):
    conn = get_connection(DBC_NAME)
    if not conn:
        return
    
    total_processed = 0
    offset = 0

    try:
        cur = conn.cursor()
        while True:
            cur.execute(
                "SELECT post_id, tokenized_text FROM public.tokenized_posts WHERE post_id > %s ORDER BY post_id ASC LIMIT %s OFFSET %s;",
                (start_id, batch_size, offset)
            )
            rows = cur.fetchall()

            if not rows:
                logging.info("✅ No more rows to train on. Training completed.")
                break

            sentences = [row[1].split() for row in rows]  # Convert space-separated text into list of words
            last_processed_id = rows[-1][0]  # Track last processed ID

            total_processed += len(rows)
            yield sentences, last_processed_id, total_processed
            offset += batch_size  # Move to next batch

    except Exception as e:
        logging.error(f"❌ Error fetching tokenized batches: {e}")

    finally:
        cur.close()
        conn.close()

# **Function to Fetch Tokenized Data in Batches**       
def fetch_tokenized_sentences(batch_size=10000, start_id=0):
    conn = get_connection(DBC_NAME)
    if not conn:
        return []

    cur = conn.cursor()
    cur.execute(
        """
        SELECT post_id, tokenized_text 
        FROM public.tokenized_posts 
        WHERE post_id > %s 
        ORDER BY post_id ASC 
        LIMIT %s;
        """,
        (start_id, batch_size),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows



# ---------------------------- TABLE CREATION (STORED IN DBC_NAME) ----------------------------

def create_stage_tables():
    """Creates all staging tables in componentsinsight (DBC_NAME)."""
    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    
    queries = [
        """
        CREATE TABLE IF NOT EXISTS stage_posts_cleaned (
            id INTEGER PRIMARY KEY,
            posttypeid INTEGER,  -- 1 = Question, 2 = Answer
            title TEXT,
            body TEXT,
            tags TEXT,
            extracted_libraries TEXT  -- Extracted library names from code blocks
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS stage_libraries_cleaned (
            id INTEGER PRIMARY KEY,
            library_name TEXT,  -- Normalized name for matching
            original_name TEXT,  -- Raw name from Libraries.io
            platform TEXT,  -- NPM, PyPI, Maven, etc.
            description TEXT  -- Processed description for NLP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS post_library_links (
            post_id INTEGER,
            library_id INTEGER,
            confidence_score FLOAT,  -- Matching confidence (0-1)
            matching_method TEXT,  -- "fuzzy", "NER", "ML"
            FOREIGN KEY (post_id) REFERENCES stage_posts_cleaned(id),
            FOREIGN KEY (library_id) REFERENCES stage_libraries_cleaned(id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tokenized_posts (
            post_id INT PRIMARY KEY,
            tokenized_text TEXT,
            tokenized_array JSONB,
            FOREIGN KEY (post_id) REFERENCES stage_posts_cleaned(id)
        );
        """,
        """
        -- Track the last processed row for training
        CREATE TABLE IF NOT EXISTS word2vec_training_progress (
            id SERIAL PRIMARY KEY,
            last_processed_id INT NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        -- Track the last processed row for training
        CREATE TABLE IF NOT EXISTS word2vec_7g_training_progress (
            id SERIAL PRIMARY KEY,
            last_processed_id INT NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """,
        """
        -- Store trained Word2Vec models
        CREATE TABLE IF NOT EXISTS word2vec_models (
            id SERIAL PRIMARY KEY,
            model_version INT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            model BYTEA NOT NULL  -- Store model as binary
        );
        """
    ]

    for query in queries:
        cur.execute(query)

    conn.commit()
    cur.close()
    conn.close()


# ---------------------------- COUNT FUNCTIONS (FROM SOURCE TABLES) ----------------------------

# **Function to Fetch Last Processed Toekn**
def last_processed_token():
    conn = get_connection(DBC_NAME)
    if not conn:
        return 0  # Return 0 if connection fails
    try:
        cur = conn.cursor()
        cur .execute("SELECT MAX(last_processed_id) FROM public.word2vec_training_progress;")
        max_id = cur.fetchone()[0] or 0  # Ensure None is converted to 0
    except Exception as e:
        print(f"❌ Error getting the last post: {e}")
        max_id = 0
    finally:
        cur.close()
        conn.close()

    return max_id  # Return the max ID or 0 if the table is empty

def last_processed_token_7g():
    conn = get_connection(DBC_NAME)
    if not conn:
        return 0  # Return 0 if connection fails
    try:
        cur = conn.cursor()
        cur .execute("SELECT MAX(last_processed_id) FROM public.word2vec_7g_training_progress;")
        max_id = cur.fetchone()[0] or 0  # Ensure None is converted to 0
    except Exception as e:
        print(f"❌ Error getting the last post: {e}")
        max_id = 0
    finally:
        cur.close()
        conn.close()

    return max_id  # Return the max ID or 0 if the table is empty

# **Function to Fetch Last Processed Post**
def last_tokenized_post():
    conn = get_connection(DBC_NAME)
    if not conn:
        return 0  # Return 0 if connection fails
    try:
        cur = conn.cursor()
        cur .execute("SELECT MAX(post_id) FROM public.tokenized_posts;")
        max_id = cur.fetchone()[0] or 0  # Ensure None is converted to 0
    except Exception as e:
        print(f"❌ Error getting the last post: {e}")
        max_id = 0
    finally:
        cur.close()
        conn.close()

    return max_id  # Return the max ID or 0 if the table is empty

def last_post():
    """
    Gets the maximum id of posts in the destination table.
    :return: Max ID of posts (int) or 0 if an error occurs.
    """

    conn = get_connection(DBC_NAME)
    if not conn:
        return 0  # Return 0 if connection fails

    try:
        cur = conn.cursor()
        cur.execute("SELECT MAX(id) FROM public.stage_posts_cleaned;")
        max_id = cur.fetchone()[0] or 0  # Ensure None is converted to 0
    except Exception as e:
        print(f"❌ Error getting the last post: {e}")
        max_id = 0
    finally:
        cur.close()
        conn.close()

    return max_id  # Return the max ID or 0 if the table is empty


def count_posts():
    """
    Count the number of posts in the Stack Overflow table.
    :return: Total number of posts (int) or None if an error occurs.
    """

    conn = get_connection(DBS_NAME)
    if not conn:
        return None

    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM public.posts_md;")
        count = cur.fetchone()[0]  # Fetch the count value
    except Exception as e:
        print(f"❌ Error counting posts: {e}")
        count = None
    finally:
        cur.close()
        conn.close()

    return count  # Return the total count

def count_libraries():
    """
    Counts total NPM packages in the Libraries.io database.
    :return: Total count of NPM packages (int) or None if an error occurs.
    """

    conn = get_connection(DBL_NAME)
    if not conn:
        return None

    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) 
            FROM Projects 
            WHERE (
                jsonb_typeof(raw) = 'object' 
                AND 
                (SELECT COUNT(*) FROM jsonb_object_keys(raw)) > 1
            )
        """)
        total_packages = cur.fetchone()[0]  # Fetch the count value
    except Exception as e:
        print(f"❌ Error counting libraries: {e}")
        total_packages = None
    finally:
        conn.close()

    return total_packages  # Return the total count


# ---------------------------- INSERTION FUNCTIONS (STORED IN DBC_NAME) ----------------------------

def insert_into_stage_posts_cleaned(data):
    """
    Inserts multiple records into stage_posts_cleaned in DBC_NAME.
    :param data: List of tuples (id, posttypeid, title, body, tags, extracted_libraries)
    """
    if not data:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    query = """
    INSERT INTO stage_posts_cleaned (id, posttypeid, title, body, tags, extracted_libraries)
    VALUES %s ON CONFLICT (id) DO NOTHING;
    """
    execute_values(cur, query, data)
    conn.commit()
    cur.close()
    conn.close()


def insert_into_stage_libraries_cleaned(data):
    """
    Inserts multiple records into stage_libraries_cleaned in DBC_NAME.
    :param data: List of tuples (id, library_name, original_name, platform, description)
    """
    if not data:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    query = """
    INSERT INTO stage_libraries_cleaned (id, library_name, original_name, platform, description)
    VALUES %s ON CONFLICT (id) DO NOTHING;
    """
    execute_values(cur, query, data)
    conn.commit()
    cur.close()
    conn.close()


def insert_into_post_library_links(data):
    """
    Inserts multiple records into post_library_links in DBC_NAME.
    :param data: List of tuples (post_id, library_id, confidence_score, matching_method)
    """
    if not data:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    query = """
    INSERT INTO post_library_links (post_id, library_id, confidence_score, matching_method)
    VALUES %s ON CONFLICT (post_id, library_id) DO NOTHING;
    """
    execute_values(cur, query, data)
    conn.commit()
    cur.close()
    conn.close()
    
def insert_into_tokenized_posts(data):
    """
    Inserts multiple records into tokenized_posts in DBC_NAME.
    :param data: List of tuples (post_id, tokenized_text, tokenized_array)
    """
    if not data:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    query = """
    INSERT INTO tokenized_posts (post_id, tokenized_text, tokenized_array)
    VALUES %s ON CONFLICT (post_id) DO NOTHING;
    """
    
    # Convert tokenized_array (list) into JSON format
    data = [(post_id, tokenized_text, json.dumps(tokenized_array)) for post_id, tokenized_text, tokenized_array in data]
    
    execute_values(cur, query, data)
    conn.commit()
    cur.close()
    conn.close()    

def update_last_processed_id(last_id):
    if not last_id:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    cur.execute("INSERT INTO public.word2vec_training_progress (last_processed_id) VALUES (%s);", (last_id,))
    conn.commit()
    cur.close()
    conn.close()  


def update_last_processed_id_7g(last_id):
    if not last_id:
        return

    conn = get_connection(DBC_NAME)
    if not conn:
        return

    cur = conn.cursor()
    cur.execute("INSERT INTO public.word2vec_7g_training_progress (last_processed_id) VALUES (%s);", (last_id,))
    conn.commit()
    cur.close()
    conn.close()  

# **Function to Save Model to Database**
def save_model_to_db(model, version):
    conn = get_connection(DBC_NAME)
    if not conn:
        return

    try:
        cur = conn.cursor()
        model_binary = pickle.dumps(model)  # Serialize model
        cur.execute(
            "INSERT INTO public.word2vec_models (model_version, model) VALUES (%s, %s);",
            (version, psycopg2.Binary(model_binary))
        )
        conn.commit()
        logging.info(f"✅ Model version {version} saved to database.")
    except Exception as e:
        logging.error(f"❌ Error saving model to DB: {e}")
    finally:
        cur.close()
        conn.close()
    
# ---------------------------- DATABASE INITIALIZATION FUNCTION ----------------------------

def initialize_staging():
    """Creates staging tables in componentsinsight (DBC_NAME)."""
    create_stage_tables()

