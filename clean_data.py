import sys
import re
import html
import markdown
from bs4 import BeautifulSoup
from database import initialize_staging
from database import read_stackoverflow_posts, insert_into_stage_posts_cleaned, count_posts
from database import read_libraries_projects, insert_into_stage_libraries_cleaned, count_libraries

# Function to clean HTML tags
def clean_html(text):
    if text:
        text = html.unescape(text)  # Decode HTML entities
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text.lower() if text else None


# Function to normalize library names
def normalize_library_name(name):
    if name:
        name = name.lower()
        name = re.sub(r'[^a-z0-9]+', '-', name)  # Replace special chars with '-'
        name = name.strip('-')
    return name

# Function to process tags field
def process_tags(tags):
    if tags:
        tags = re.sub(r'<|>', ' ', tags)  # Replace '<>' with space
        tags = re.split(r'\s+', tags.strip())  # Split into a list
        return ', '.join(tags)  # Convert back to comma-separated string
    return None

# Function to remove @mentions
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

# Function to clean markdown and extract meaningful text
def clean_markdown(text):
    if not text:
        return None

    # Convert Markdown to HTML
    html_content = markdown.markdown(text)

    # Parse HTML with BeautifulSoup to remove links, images, unnecessary formatting
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove links, images, blockquotes
    for tag in soup(["a", "img", "blockquote"]):
        tag.extract()

    # Extract text
    clean_text = soup.get_text()

    # Normalize spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return remove_mentions(clean_text.lower())


# Function to extract possible library mentions from code
def extract_libraries_from_code(text):
    if not text:
        return None

    # Extract meaningful library-related lines (like `pip install xyz` or `import xyz`)
    matches = re.findall(r'\b(?:pip install|npm install|import|require|from)\s+([\w-]+)', text, re.IGNORECASE)

    return ', '.join(set(matches)) if matches else None

# Function to clean StackOverflow posts
def clean_stackoverflow_posts(batch_size=10000):
    """
    Cleans Stack Overflow posts in batches and stores them in stage_posts_cleaned.
    - Reads posts in paginated batches.
    - Cleans the markdown content, extracts libraries, and processes tags.
    - Inserts cleaned data into the staging table.
    """
    total_posts = count_posts()
    if total_posts is None:
        print("❌ Error: Could not retrieve post count.")
        return
    
    print(f"🔄 Processing {total_posts} Stack Overflow posts in batches of {batch_size}...")

    processed_count = 0

    for batch in read_stackoverflow_posts(batch_size=batch_size):
        cleaned_data = []
        for post_id, post_type, title, body, tags in batch:
            cleaned_body = clean_markdown(body)  # Clean markdown content
            extracted_libraries = extract_libraries_from_code(body)  # Extract possible libraries

            cleaned_data.append((
                post_id,
                post_type,
                clean_markdown(title),  # Clean title
                cleaned_body,
                process_tags(tags),
                extracted_libraries  # Extracted libraries from code blocks
            ))

        # Insert cleaned batch into stage_posts_cleaned
        insert_into_stage_posts_cleaned(cleaned_data)

        processed_count += len(cleaned_data)
        print(f"✅ Processed {processed_count}/{total_posts} posts...")

    print(f"🎉 Finished processing all {processed_count} Stack Overflow posts!")


def clean_libraries_projects(batch_size=10000):
    """
    Cleans Libraries.io projects in batches and stores them in stage_libraries_cleaned.
    - Reads projects in paginated batches.
    - Normalizes library names and cleans descriptions.
    - Inserts cleaned data into the staging table.
    """
    total_libraries = count_libraries()
    if total_libraries is None:
        print("❌ Error: Could not retrieve project count.")
        return

    print(f"🔄 Processing {total_libraries} Libraries.io projects in batches of {batch_size}...")

    processed_count = 0

    for batch in read_libraries_projects(batch_size=batch_size):
        cleaned_data = []
        for id, name, platform, description in batch:
            cleaned_data.append((
                id,
                normalize_library_name(name),  # Normalize library name
                name,  # Keep the original name
                platform,
                clean_markdown(description) #clean_html(description)  # Clean description text
            ))

        # Insert cleaned batch into stage_libraries_cleaned
        insert_into_stage_libraries_cleaned(cleaned_data)

        processed_count += len(cleaned_data)
        print(f"✅ Processed {processed_count}/{total_libraries} projects...")

    print(f"🎉 Finished processing all {processed_count} Libraries.io projects!")
    
# The main function
def main():
    """
    Reads command-line input to determine which dataset to clean.
    Usage:
        python clean_data.py 1   -> Cleans Stack Overflow posts
        python clean_data.py 2   -> Cleans Libraries.io projects
        python clean_data.py all -> Cleans both datasets
    """
    if len(sys.argv) != 2:
        print("❌ Invalid usage. Please provide an option: \n"
              "1 - Clean Stack Overflow posts\n"
              "2 - Clean Libraries.io projects\n"
              "all - Clean both datasets")
        sys.exit(1)
    
    initialize_staging()
    
    option = sys.argv[1].lower()

    if option == "1":
        print("🔄 Starting Stack Overflow post cleaning...")
        clean_stackoverflow_posts(batch_size=5000)
    elif option == "2":
        print("🔄 Starting Libraries.io project cleaning...")
        clean_libraries_projects(batch_size=5000)
    elif option == "all":
        print("🔄 Cleaning both datasets...")
        clean_stackoverflow_posts(batch_size=5000)
        clean_libraries_projects(batch_size=5000)
    else:
        print("❌ Invalid option. Use '1', '2', or 'all'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
