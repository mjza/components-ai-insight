import sys
import re
import html
import markdown
from tqdm import tqdm
from bs4 import BeautifulSoup
from database import initialize_staging
from database import read_stackoverflow_posts, insert_into_stage_posts_cleaned, count_posts, last_post
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

# Function to remove @mentions
def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

# Function to clean markdown and extract meaningful text
def clean_markdown(text, preserve_inline_code=True):
    """
    Cleans Markdown text by:
    - Removing multi-line code blocks (```code```)
    - Removing links and images
    - Optionally keeping or removing inline code (`code`)
    - Removing all other Markdown formatting
    :param text: Markdown-formatted text
    :param preserve_inline_code: If True, keeps text inside single backticks (`example`)
    :return: Cleaned text without Markdown formatting
    """
    if not text:
        return None

    # Remove multi-line code blocks (```code```)
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove links (e.g., [text](http://example.com))
    text = re.sub(r'\[([^\]]+)\]\((https?:\/\/[^\s]+)\)', r'\1', text)  # Keep link text, remove URL

    # Remove images (e.g., ![alt](image.jpg))
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)

    # Handle inline code (`example`)
    if preserve_inline_code:
        text = re.sub(r'`([^`]*)`', r'\1', text)  # Remove backticks but keep text inside
    else:
        text = re.sub(r'`([^`]*)`', '', text)  # Remove inline code   

    # Remove remaining Markdown formatting (bold, italic, headers, lists)
    text = re.sub(r'[#*_>~-]+', '', text)  # Remove #, *, _, >, ~, -, etc.

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = remove_mentions(text.lower()) 

    return text


def extract_libraries_from_code(text):
    """
    Extracts possible library mentions from Markdown code blocks.
    - Captures library names from import, require, and package manager install commands.
    - Detect C/C++ headers (#include <sys/socket.h>).
    :param text: Markdown-formatted body text
    :return: Comma-separated string of extracted libraries, or None if nothing is found
    """
    if not text:
        return None

    # List of programming variable names to ignore
    COMMON_VARIABLES = {"i", "x", "y", "z", "data", "query", "item", "row", "col", "temp", "val"}

    # Dynamically generate regex patterns from package manager commands
    PACKAGE_MANAGERS = {
        "PureScript": ["pulp dep install"],
        "Objective-C": ["pod install", "carthage update"],
        "C++": ["vcpkg install", "conan install"],
        "JavaScript": ["npm install", "yarn add", "bower install", "meteor add"],
        "Java": ["mvn install", "gradle dependencies"],
        "Python": ["pip install", "conda install"],
        "C#": ["dotnet add package", "nuget install"],
        "PHP": ["composer require"],
        "Ruby": ["gem install", "bundle add"],
        "Rust": ["cargo install", "cargo add"],
        "CSS": ["bower install"],
        "Dart": ["pub add"],
        "Perl": ["cpan install"],
        "R": ["install.packages"],
        "Clojure": ["clojure -Sdeps"],
        "Elixir": ["mix deps.get"],
        "C": ["vcpkg install", "conan install"],
        "Puppet": ["puppet module install"],
        "Swift": ["swift package add", "pod install", "carthage update"],
        "Julia": ["Pkg.add"],
        "Elm": ["elm install"],
        "D": ["dub add"],
        "Nim": ["nimble install"],
        "Haxe": ["haxelib install"],
        "Go": ["go get"]
    }

    # Generate install command regex from package managers
    INSTALL_COMMANDS = [cmd for commands in PACKAGE_MANAGERS.values() for cmd in commands]
    INSTALL_REGEX = r'\b(?:' + '|'.join(re.escape(cmd) for cmd in INSTALL_COMMANDS) + r')\s+["\']?([a-zA-Z0-9._-]+)["\']?'

    # Stricter regex for standard import statements
    IMPORT_REGEX = r'^\s*(?:import|require|include|using)\s+["\'<]?([a-zA-Z0-9._-]+)["\'>]?'

    # Improved "from x import y" rule for Python-style imports
    PYTHON_FROM_IMPORT_REGEX = r'^\s*from\s+([a-zA-Z0-9._-]+)\s+import\s+'

    # Extract content inside inline (`code`) and block (```code```) backticks
    code_blocks = re.findall(r'`{1,3}(.*?)`{1,3}', text, re.DOTALL)

    libraries = set()
    for block in code_blocks:

        # Find matches using dynamically generated regex
        matches = (
            re.findall(INSTALL_REGEX, block, re.IGNORECASE) +
            re.findall(IMPORT_REGEX, block, re.IGNORECASE) +
            re.findall(PYTHON_FROM_IMPORT_REGEX, block, re.IGNORECASE)
        )

        for match in matches:
            # Ensure it's not a common variable name
            if match not in COMMON_VARIABLES:
                libraries.add(match)

    return ', '.join(libraries) if libraries else None




# Function to clean StackOverflow posts with progress bar
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

    start_id = last_post()

    processed_count = 0

    # Initialize the progress bar
    with tqdm(total=total_posts, desc="Processing Posts", unit="post") as pbar:
        for batch in read_stackoverflow_posts(batch_size=batch_size, start_id=start_id):
            cleaned_data = []
            for post_id, post_type, title, body, tags in batch:
                cleaned_body = clean_markdown(body)  # Clean markdown content
                extracted_libraries = extract_libraries_from_code(body)  # Extract possible libraries
                processed_tags = tags.strip().lower() if tags else None
                
                cleaned_data.append((
                    post_id,
                    post_type,
                    clean_markdown(title),  # Clean title
                    cleaned_body,
                    processed_tags,
                    extracted_libraries  # Extracted libraries from code blocks
                ))

            # Insert cleaned batch into stage_posts_cleaned
            insert_into_stage_posts_cleaned(cleaned_data)

            processed_count += len(cleaned_data)
            pbar.update(len(cleaned_data))  # Update the progress bar

    print(f"🎉 Finished processing all {processed_count} Stack Overflow posts!")


# Function to clean Libraries.io projects with a progress bar
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

    # Initialize the progress bar
    with tqdm(total=total_libraries, desc="Processing Libraries", unit="project") as pbar:
        for batch in read_libraries_projects(batch_size=batch_size):
            cleaned_data = []
            for id, name, platform, description in batch:
                cleaned_data.append((
                    id,
                    normalize_library_name(name),  # Normalize library name
                    name,  # Keep the original name
                    platform,
                    clean_markdown(description)  # Clean description text
                ))

            # Insert cleaned batch into stage_libraries_cleaned
            insert_into_stage_libraries_cleaned(cleaned_data)

            processed_count += len(cleaned_data)
            pbar.update(len(cleaned_data))  # Update the progress bar

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
